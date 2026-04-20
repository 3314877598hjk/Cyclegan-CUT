"""
推理性能基准测试脚本。

测试内容：
  1. 单任务单张图像平均推理延迟（warm-up 后计时）
  2. GPU 显存峰值占用
  3. 多任务切换时 LRU 缓存 vs 无缓存的加载耗时对比
  4. 连续推理稳定性（100 张图像批量推理耗时）

用法：
  python scripts/benchmark_inference.py --task "Map -> Vector" --n_warmup 5 --n_runs 50
  python scripts/benchmark_inference.py --benchmark_switch  # 测试多任务切换
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

# 将项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_random_image(size: int = 256) -> Image.Image:
    import numpy as np
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def _gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _gpu_memory_reserved_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 ** 2
    return 0.0


TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# --------------------------------------------------------------------------- #
# 单任务推理延迟基准
# --------------------------------------------------------------------------- #

def benchmark_single_task(task_name: str, n_warmup: int = 5, n_runs: int = 50):
    from app import load_model, TASKS

    if task_name not in TASKS:
        raise ValueError(f"未知任务: {task_name}。可选: {list(TASKS.keys())}")

    print(f"\n{'='*60}")
    print(f"  单任务推理基准：{task_name}")
    print(f"  Warm-up: {n_warmup} 次 | 正式计时: {n_runs} 次")
    print(f"{'='*60}")

    model, opt = load_model(task_name)
    device = opt.device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = _gpu_memory_mb()

    # Warm-up
    for _ in range(n_warmup):
        img = _make_random_image()
        t = TRANSFORM(img).unsqueeze(0).to(device)
        data = {"A": t, "B": t, "A_paths": [""], "B_paths": [""]}
        with torch.no_grad():
            model.set_input(data)
            model.test()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    mem_after_load = _gpu_memory_mb()

    # 正式计时
    latencies = []
    for _ in range(n_runs):
        img = _make_random_image()
        t = TRANSFORM(img).unsqueeze(0).to(device)
        data = {"A": t, "B": t, "A_paths": [""], "B_paths": [""]}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model.set_input(data)
            model.test()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    import statistics
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

    print(f"\n  推理延迟（ms）")
    print(f"    平均值:  {statistics.mean(latencies):.2f}")
    print(f"    中位数:  {statistics.median(latencies):.2f}")
    print(f"    最小值:  {min(latencies):.2f}")
    print(f"    最大值:  {max(latencies):.2f}")
    print(f"    标准差:  {statistics.stdev(latencies):.2f}")
    print(f"\n  GPU 显存（MB）")
    print(f"    模型加载前:  {mem_before:.1f}")
    print(f"    模型加载后:  {mem_after_load:.1f}")
    print(f"    峰值占用:    {peak_mem:.1f}")
    print(f"    模型占用增量: {mem_after_load - mem_before:.1f}")

    return {
        "task": task_name,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_std_ms": statistics.stdev(latencies),
        "gpu_peak_mb": peak_mem,
        "gpu_model_mb": mem_after_load - mem_before,
    }


# --------------------------------------------------------------------------- #
# 多任务切换基准
# --------------------------------------------------------------------------- #

def benchmark_task_switching():
    from app import load_model, TASKS, model_manager

    tasks = list(TASKS.keys())
    print(f"\n{'='*60}")
    print(f"  多任务切换基准（LRU 缓存容量=1）")
    print(f"  任务列表: {tasks}")
    print(f"{'='*60}")

    switch_times = []
    for i, task in enumerate(tasks * 2):  # 来回切换两轮
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        load_model(task)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        action = "命中缓存" if elapsed_ms < 50 else "加载新模型"
        print(f"  [{i+1}] {task:<25} → {action}  ({elapsed_ms:.0f} ms)")
        switch_times.append((task, elapsed_ms))

    cold_times = [t for _, t in switch_times if t >= 50]
    cache_times = [t for _, t in switch_times if t < 50]

    print(f"\n  冷加载平均耗时:  {sum(cold_times)/len(cold_times):.0f} ms" if cold_times else "")
    print(f"  缓存命中平均耗时: {sum(cache_times)/len(cache_times):.0f} ms" if cache_times else "")


# --------------------------------------------------------------------------- #
# 连续推理稳定性
# --------------------------------------------------------------------------- #

def benchmark_throughput(task_name: str, n_images: int = 100):
    from app import load_model, TASKS

    print(f"\n{'='*60}")
    print(f"  吞吐量测试：连续推理 {n_images} 张图像")
    print(f"  任务: {task_name}")
    print(f"{'='*60}")

    model, opt = load_model(task_name)
    device = opt.device

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in range(n_images):
        img = _make_random_image()
        t = TRANSFORM(img).unsqueeze(0).to(device)
        data = {"A": t, "B": t, "A_paths": [""], "B_paths": [""]}
        with torch.no_grad():
            model.set_input(data)
            model.test()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - t_start

    fps = n_images / total_sec
    print(f"\n  总耗时: {total_sec:.2f}s")
    print(f"  吞吐量: {fps:.1f} FPS  ({1000/fps:.1f} ms/张)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="推理性能基准测试")
    parser.add_argument("--task", type=str, default="Map -> Vector",
                        help="要测试的任务名称")
    parser.add_argument("--n_warmup", type=int, default=5, help="Warm-up 次数")
    parser.add_argument("--n_runs", type=int, default=50, help="计时推理次数")
    parser.add_argument("--benchmark_switch", action="store_true",
                        help="测试多任务切换性能")
    parser.add_argument("--benchmark_throughput", action="store_true",
                        help="测试连续推理吞吐量")
    parser.add_argument("--n_images", type=int, default=100, help="吞吐量测试图像数量")
    return parser.parse_args()


def main():
    args = parse_args()

    device_info = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"\n运行设备: {device_info}")

    if args.benchmark_switch:
        benchmark_task_switching()
    elif args.benchmark_throughput:
        benchmark_throughput(args.task, args.n_images)
    else:
        benchmark_single_task(args.task, args.n_warmup, args.n_runs)


if __name__ == "__main__":
    main()
