import collections
import gc
import torch
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image

from models import create_model
from util import util

# =====================================================
# 1️⃣ 任务配置（多任务核心）
# =====================================================
TASKS = {
    "🗺️ Map → 矢量图": {
        "name": "map2vector_cyclegan",
        "direction": "AtoB"
    },
    "   矢量图 → Map":{
        "name": "map2vector_cyclegan",
        "direction":"BtoA"
    },
    "🐎 马 → 斑马": {
        "name": "zebra2horse_cyclegan",
        "direction": "AtoB"
    },
    "🦓 斑马 → 马": {
        "name": "zebra2horse_cyclegan",
        "direction": "BtoA"
    }
}

# =====================================================
# 2️⃣ CycleGAN 推理配置
# =====================================================
class Opt:
    def __init__(self, name, direction):
        self.gpu_ids = [0]
        self.name = name
        self.checkpoints_dir = "./checkpoints"
        self.model = "cycle_gan"
        self.netG = "resnet_9blocks"
        self.norm = "instance"
        self.no_dropout = True
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.load_size = 256
        self.crop_size = 256
        self.preprocess = "resize_and_crop"
        self.serial_batches = True
        self.display_winsize = 256
        self.isTrain = False
        self.direction = direction
        self.epoch = "latest"
        self.num_test = float("inf")
        self.phase = "test"
        self.batch_size = 1
        self.verbose = False
        self.init_type = "normal"
        self.init_gain = 0.02
        self.load_iter = 0

        self.device = torch.device("cuda:0" if self.gpu_ids else "cpu")


# =====================================================
# 3️⃣ 模型缓存（工程级优化：LRU + 显存动态释放）
# =====================================================
class LRUModelCache:
    def __init__(self, max_capacity=1):
        """
        max_capacity: 允许同时驻留在 GPU 显存中的最大模型数量。
        如果你的显卡显存较小（比如 8GB 以下），建议设为 1；
        如果显存充裕，可以设为 2 或更大，切换会更流畅。
        """
        self.max_capacity = max_capacity
        self.cache = collections.OrderedDict()

    def get(self, task_name):
        # 场景 A：模型已经在缓存中（缓存命中）
        if task_name in self.cache:
            # 将该任务移到字典末尾，标记为“最近刚刚使用过”
            self.cache.move_to_end(task_name)
            print(f"[INFO] ⚡ 缓存命中，直接使用模型: {task_name}")
            return self.cache[task_name]

        # 场景 B：模型不在缓存中，且缓存已达到上限，需要踢出最旧的模型
        if len(self.cache) >= self.max_capacity:
            # 弹出最早放入且最近没用过的模型 (last=False 表示从头部弹出)
            oldest_task, (old_model, _) = self.cache.popitem(last=False)
            print(f"[INFO] 🧹 缓存已满，正在卸载旧模型以释放显存: {oldest_task}")

            # 核心显存释放逻辑
            try:
                # 1. 如果模型是分布式的或者有 module 属性，尝试获取原生模型
                net = old_model.module if hasattr(old_model, 'module') else old_model
                # 2. 将参数强制转移到 CPU
                if hasattr(net, 'cpu'):
                    net.cpu()
                # 3. 删除 Python 引用
                del old_model
                del net
            except Exception as e:
                print(f"[WARNING] 卸载模型时出现小问题: {e}")

            # 4. 强制 Python 垃圾回收并清空 PyTorch 的 CUDA 缓存碎片
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 场景 C：加载新模型并放入缓存
        print(f"[INFO] 🚀 正在加载新模型至显存: {task_name}")
        cfg = TASKS[task_name]
        opt = Opt(cfg["name"], cfg["direction"])

        # 动态创建并设置模型
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        # 存入缓存
        self.cache[task_name] = (model, opt)
        return self.cache[task_name]


# 实例化全局缓存管理器
model_manager = LRUModelCache(max_capacity=1)


def load_model(task_name):
    """
    统一的模型获取接口，替代原有的全局字典
    """
    return model_manager.get(task_name)


# =====================================================
# 4️⃣ 图像预处理
# =====================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# =====================================================
# 5️⃣ 推理函数
# =====================================================
def inference(input_image, task_name):
    model, opt = load_model(task_name)

    with torch.no_grad():
        input_image = input_image.convert("RGB")
        img_tensor = transform(input_image).unsqueeze(0).to(opt.device)

        data = {
            "A": img_tensor,
            "B": img_tensor,
            "A_paths": ["input.png"],
            "B_paths": ["input.png"]
        }

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        if opt.direction == "AtoB":
            output = visuals["fake_B"]
        else:
            output = visuals["fake_A"]

        return util.tensor2im(output)


# =====================================================
# 6️⃣ Gradio Demo
# =====================================================
demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="pil", label="上传输入图像"),
        gr.Dropdown(
            choices=list(TASKS.keys()),
            value="🗺️ Map → 矢量图",
            label="选择迁移任务"
        )
    ],
    outputs=gr.Image(type="numpy", label="生成结果"),
    title="🎓 基于 CycleGAN 的多任务图像迁移毕业设计 Demo",
    description=(
        "支持多种无配对图像迁移任务，如 Map→矢量图、斑马↔马。\n"
        "系统采用统一推理框架与模型缓存机制，实现多模型高效调度。"
    )
)

# =====================================================
# 7️⃣ 启动
# =====================================================
if __name__ == "__main__":
    demo.launch(share=True)
