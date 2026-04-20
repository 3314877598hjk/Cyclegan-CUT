"""
评估脚本：计算生成图像与真实图像之间的图像质量指标。

支持以下指标：
  - SSIM  (结构相似性，via scikit-image)
  - LPIPS (感知相似性，via lpips)
  - FID   (Frechet Inception Distance，via torch_fidelity)
  - 边缘保持率 (Sobel 边缘 L1 距离)

用法示例（对消融实验各组进行评估）：

  # G0: 基线
  python scripts/eval_metrics.py \\
      --real_dir datasets/maps/testB \\
      --fake_dir results/baseline_cyclegan/test_latest/images/fake_B

  # G3: 全改进模型
  python scripts/eval_metrics.py \\
      --real_dir datasets/maps/testB \\
      --fake_dir results/improved_cyclegan/test_latest/images/fake_B \\
      --compute_fid
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_image_paths(directory: str) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not paths:
        raise FileNotFoundError(f"目录中未找到图像文件: {directory}")
    return paths


def _load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """HWC numpy [0,1] → CHW tensor [0,1]"""
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def _sobel_edges(img_np: np.ndarray) -> np.ndarray:
    """Simple Sobel edge map from HWC [0,1] RGB image."""
    gray = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
    from scipy.ndimage import convolve

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    ex = convolve(gray, kx)
    ey = convolve(gray, ky)
    return np.sqrt(ex ** 2 + ey ** 2 + 1e-6)


# --------------------------------------------------------------------------- #
# Metric: SSIM
# --------------------------------------------------------------------------- #

def compute_ssim(real_paths: list[Path], fake_paths: list[Path], size: int = 256) -> dict:
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("[SSIM] 跳过：未安装 scikit-image。请运行 pip install scikit-image")
        return {}

    scores = []
    for rp, fp in zip(real_paths, fake_paths):
        r = _load_rgb(rp, size)
        f = _load_rgb(fp, size)
        score = ssim(r, f, data_range=1.0, channel_axis=2)
        scores.append(score)

    return {"SSIM_mean": float(np.mean(scores)), "SSIM_std": float(np.std(scores))}


# --------------------------------------------------------------------------- #
# Metric: LPIPS
# --------------------------------------------------------------------------- #

def compute_lpips(real_paths: list[Path], fake_paths: list[Path], size: int = 256) -> dict:
    try:
        import lpips
    except ImportError:
        print("[LPIPS] 跳过：未安装 lpips。请运行 pip install lpips")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    scores = []
    with torch.no_grad():
        for rp, fp in zip(real_paths, fake_paths):
            r = _to_tensor(_load_rgb(rp, size)).to(device) * 2 - 1  # [0,1] → [-1,1]
            f = _to_tensor(_load_rgb(fp, size)).to(device) * 2 - 1
            score = loss_fn(r, f).item()
            scores.append(score)

    return {"LPIPS_mean": float(np.mean(scores)), "LPIPS_std": float(np.std(scores))}


# --------------------------------------------------------------------------- #
# Metric: FID
# --------------------------------------------------------------------------- #

def compute_fid(real_dir: str, fake_dir: str) -> dict:
    try:
        import torch_fidelity
    except ImportError:
        print("[FID] 跳过：未安装 torch_fidelity。请运行 pip install torch-fidelity")
        return {}

    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=False,
    )
    return {"FID": metrics.get("frechet_inception_distance", None)}


# --------------------------------------------------------------------------- #
# Metric: Edge Preservation
# --------------------------------------------------------------------------- #

def compute_edge_preservation(real_paths: list[Path], fake_paths: list[Path], size: int = 256) -> dict:
    try:
        from scipy.ndimage import convolve  # noqa: F401
    except ImportError:
        print("[Edge] 跳过：未安装 scipy。请运行 pip install scipy")
        return {}

    scores = []
    for rp, fp in zip(real_paths, fake_paths):
        r = _load_rgb(rp, size)
        f = _load_rgb(fp, size)
        er = _sobel_edges(r)
        ef = _sobel_edges(f)
        # Normalize edge maps
        er = er / (er.max() + 1e-8)
        ef = ef / (ef.max() + 1e-8)
        score = 1.0 - float(np.mean(np.abs(er - ef)))  # 越高越好
        scores.append(score)

    return {"EdgePreservation_mean": float(np.mean(scores)), "EdgePreservation_std": float(np.std(scores))}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="图像翻译质量评估脚本")
    parser.add_argument("--real_dir", required=True, help="真实目标域图像目录")
    parser.add_argument("--fake_dir", required=True, help="生成图像目录")
    parser.add_argument("--size", type=int, default=256, help="评估时统一缩放到的分辨率")
    parser.add_argument("--compute_fid", action="store_true", help="计算 FID（较慢，需要 torch-fidelity）")
    parser.add_argument("--max_images", type=int, default=None, help="最多评估多少张图像（None=全部）")
    return parser.parse_args()


def main():
    args = parse_args()

    real_paths = _load_image_paths(args.real_dir)
    fake_paths = _load_image_paths(args.fake_dir)

    n_real, n_fake = len(real_paths), len(fake_paths)
    if n_real != n_fake:
        print(f"[警告] 真实图像数量 ({n_real}) 与生成图像数量 ({n_fake}) 不一致，取最小值。")
    n = min(n_real, n_fake)
    if args.max_images:
        n = min(n, args.max_images)
    real_paths = real_paths[:n]
    fake_paths = fake_paths[:n]

    print(f"\n{'='*60}")
    print(f"  评估目录：{args.fake_dir}")
    print(f"  图像数量：{n}")
    print(f"  分辨率：  {args.size}x{args.size}")
    print(f"{'='*60}\n")

    results = {}
    t0 = time.time()

    results.update(compute_ssim(real_paths, fake_paths, args.size))
    results.update(compute_lpips(real_paths, fake_paths, args.size))
    results.update(compute_edge_preservation(real_paths, fake_paths, args.size))

    if args.compute_fid:
        results.update(compute_fid(args.real_dir, args.fake_dir))

    elapsed = time.time() - t0
    print(f"{'指标':<30} {'值':>10}")
    print("-" * 42)
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<28} {v:>10.4f}")
    print(f"\n  评估耗时: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
