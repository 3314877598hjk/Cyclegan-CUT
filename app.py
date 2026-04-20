from __future__ import annotations

import collections
import gc
from dataclasses import dataclass, field

import torch
import torchvision.transforms as transforms

from models import create_model
from util import util


TASKS = {
    "Map -> Vector": {
        "name": "map2vector_cyclegan",
        "direction": "AtoB",
    },
    "Vector -> Map": {
        "name": "map2vector_cyclegan",
        "direction": "BtoA",
    },
    "Horse -> Zebra": {
        "name": "zebra2horse_cyclegan",
        "direction": "AtoB",
    },
    "Zebra -> Horse": {
        "name": "zebra2horse_cyclegan",
        "direction": "BtoA",
    },
}


@dataclass
class InferenceOptions:
    name: str
    direction: str
    checkpoints_dir: str = "./checkpoints"
    model: str = "cycle_gan"
    netG: str = "resnet_9blocks"
    norm: str = "instance"
    no_dropout: bool = True
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    load_size: int = 256
    crop_size: int = 256
    preprocess: str = "resize_and_crop"
    serial_batches: bool = True
    display_winsize: int = 256
    isTrain: bool = False
    epoch: str = "latest"
    phase: str = "test"
    batch_size: int = 1
    verbose: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02
    load_iter: int = 0
    continue_train: bool = False
    no_attention: bool = False  # existing checkpoints were trained with attention enabled
    gpu_ids: list[int] = field(default_factory=list)
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.device = torch.device("cuda:0" if self.gpu_ids else "cpu")


class LRUModelCache:
    def __init__(self, max_capacity: int = 1):
        self.max_capacity = max_capacity
        self.cache: collections.OrderedDict[str, tuple[object, InferenceOptions]] = collections.OrderedDict()

    def _release_model(self, model: object) -> None:
        for name in getattr(model, "model_names", []):
            if not isinstance(name, str):
                continue
            net = getattr(model, "net" + name, None)
            if hasattr(net, "cpu"):
                net.cpu()

    def get(self, task_name: str):
        if task_name in self.cache:
            self.cache.move_to_end(task_name)
            print(f"[INFO] Reusing cached model for task: {task_name}")
            return self.cache[task_name]

        if len(self.cache) >= self.max_capacity:
            oldest_task, (old_model, _) = self.cache.popitem(last=False)
            print(f"[INFO] Evicting cached model for task: {oldest_task}")
            self._release_model(old_model)
            del old_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg = TASKS[task_name]
        opt = InferenceOptions(cfg["name"], cfg["direction"])
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        self.cache[task_name] = (model, opt)
        return self.cache[task_name]


model_manager = LRUModelCache(max_capacity=1)


def load_model(task_name: str):
    return model_manager.get(task_name)


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def inference(input_image, task_name: str):
    if input_image is None:
        raise ValueError("Please upload an image before running inference.")

    model, opt = load_model(task_name)

    with torch.no_grad():
        input_image = input_image.convert("RGB")
        img_tensor = transform(input_image).unsqueeze(0).to(opt.device)

        data = {
            "A": img_tensor,
            "B": img_tensor,
            "A_paths": ["input.png"],
            "B_paths": ["input.png"],
        }

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        output = visuals["fake_B"] if opt.direction == "AtoB" else visuals["fake_A"]
        return util.tensor2im(output)


def build_demo():
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Gradio could not be imported. Install or repair the demo dependencies, "
            "then try again. Suggested packages: gradio, setuptools."
        ) from exc

    return gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(type="pil", label="输入图像 / Input Image"),
            gr.Dropdown(
                choices=list(TASKS.keys()),
                value="Map -> Vector",
                label="翻译任务 / Translation Task",
            ),
        ],
        outputs=gr.Image(type="numpy", label="输出图像 / Output Image"),
        title="基于改进 CycleGAN 的无配对图像翻译演示系统",
        description=(
            "本系统基于 CycleGAN 框架，集成了自注意力模块（Self-Attention）与 Sobel 边缘一致性损失，"
            "支持多任务切换与 LRU 显存调度。当前支持地图矢量化与马斑马风格迁移任务。\n\n"
            "Unpaired image-to-image translation demo with self-attention generator and edge consistency loss. "
            "Supports multi-task LRU model caching for efficient GPU memory management."
        ),
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
