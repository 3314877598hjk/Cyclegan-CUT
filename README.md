<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN and pix2pix in PyTorch

**Udpate in 2025**: we recently updated the code to support Python 3.11 and PyTorch 2.4. It also supports DDP for single-machine multiple-GPU training. (Please use `torchrun --nproc_per_node=4 train.py ...`)

**New**: Please check out [img2img-turbo](https://github.com/GaParmar/img2img-turbo) repo that includes both pix2pix-turbo and CycleGAN-Turbo. Our new one-step image-to-image translation methods can support both paired and unpaired training and produce better results by leveraging the pre-trained StableDiffusion-Turbo model. The inference time for 512x512 image is 0.29 sec on A6000 and 0.11 sec on A100.

Please check out [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT), our new unpaired image-to-image translation model that enables fast and memory-efficient training.

We provide PyTorch implementations for both unpaired and paired image-to-image translation.

The code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesungp), and supported by [Tongzhou Wang](https://github.com/SsnL).

This PyTorch implementation produces results comparable to or better than our original Torch software. If you would like to reproduce the same results as in the papers, check out the original [CycleGAN Torch](https://github.com/junyanz/CycleGAN) and [pix2pix Torch](https://github.com/phillipi/pix2pix) code in Lua/Torch.

**Note**: The current software works well with PyTorch 2.4+. Check out the older [branch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1) that supports PyTorch 0.1-0.3.

You may find useful information in [training/test tips](docs/tips.md) and [frequently asked questions](docs/qa.md). To implement custom models and datasets, check out our [templates](#custom-model-and-dataset). To help users better understand and adapt our codebase, we provide an [overview](docs/overview.md) of the code structure of this repository.

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) | [Paper](https://arxiv.org/pdf/1703.10593.pdf) | [Torch](https://github.com/junyanz/CycleGAN) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>

**Pix2pix: [Project](https://phillipi.github.io/pix2pix/) | [Paper](https://arxiv.org/pdf/1611.07004.pdf) | [Torch](https://github.com/phillipi/pix2pix) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>

**[EdgesCats Demo](https://affinelayer.com/pixsrv/) | [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) | by [Christopher Hesse](https://twitter.com/christophrhesse)**

<img src='imgs/edges2cats.jpg' width="400px"/>

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*, [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (\* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)

## Talks and Course

pix2pix slides: [keynote](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.key) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf),
CycleGAN slides: [pptx](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pptx) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pdf)

CycleGAN course assignment [code](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip) and [handout](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) designed by Prof. [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) for [CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" at University of Toronto. Please contact the instructor if you would like to adopt it in your course.

## Colab Notebook

TensorFlow Core CycleGAN Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

TensorFlow Core pix2pix Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)

PyTorch Colab notebook: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) and [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

ZeroCostDL4Mic Colab notebook: [CycleGAN](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/CycleGAN_ZeroCostDL4Mic.ipynb) and [pix2pix](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/pix2pix_ZeroCostDL4Mic.ipynb)

## Other implementations

### CycleGAN

<p><a href="https://github.com/leehomyc/cyclegan-1"> [Tensorflow]</a> (by Harry Yang),
<a href="https://github.com/architrathore/CycleGAN/">[Tensorflow]</a> (by Archit Rathore),
<a href="https://github.com/vanhuyz/CycleGAN-TensorFlow">[Tensorflow]</a> (by Van Huy),
<a href="https://github.com/XHUJOY/CycleGAN-tensorflow">[Tensorflow]</a> (by Xiaowei Hu),
<a href="https://github.com/LynnHo/CycleGAN-Tensorflow-2"> [Tensorflow2]</a> (by Zhenliang He),
<a href="https://github.com/luoxier/CycleGAN_Tensorlayer"> [TensorLayer1.0]</a> (by luoxier),
<a href="https://github.com/tensorlayer/cyclegan"> [TensorLayer2.0]</a> (by zsdonghao),
<a href="https://github.com/Aixile/chainer-cyclegan">[Chainer]</a> (by Yanghua Jin),
<a href="https://github.com/yunjey/mnist-svhn-transfer">[Minimal PyTorch]</a> (by yunjey),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/CycleGAN">[Mxnet]</a> (by Ldpe2G),
<a href="https://github.com/tjwei/GANotebooks">[lasagne/Keras]</a> (by tjwei),
<a href="https://github.com/simontomaskarlsson/CycleGAN-Keras">[Keras]</a> (by Simon Karlsson),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Oneflow-Python/CycleGAN">[OneFlow]</a> (by Ldpe2G)
</p>
</ul>

### pix2pix

<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>
</ul>

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and other dependencies. For Conda users, you can create a new Conda environment by

```bash
conda env create -f environment.yml
```

and then activate the environment by

```bash
conda activate pytorch-img2img
```

- For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
- For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test

- Download a CycleGAN dataset (e.g. maps):

```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```

- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with training script
- Train a model:

```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name map2vector_cyclegan --model cycle_gan --use_wandb
```

To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.

- Test the model:

```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name map2vector_cyclegan --model cycle_gan
```

- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test

- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):

```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with training script
- Train a model:

```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA  --use_wandb
```

To see more intermediate results, check out `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):

```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.

### Apply a pre-trained model (CycleGAN)

- You can download a pretrained model (e.g. horse2zebra) with the following script:

```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```

- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the horse2zebra dataset:

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using

```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```

- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For pix2pix and your own models, you need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. See this [FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296) for more details.

### Apply a pre-trained model (pix2pix)

Download a pre-trained model with `./scripts/download_pix2pix_model.sh`.

- Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,

```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```

- Download the pix2pix facades datasets:

```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

- Then generate the results using

```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```

- Note that we specified `--direction BtoA` as Facades dataset's A to B direction is photos to labels.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--model test` option. See `./scripts/test_single.sh` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).

- See a list of currently available models at `./scripts/download_pix2pix_model.sh`

### Multi-GPU training

To train a model on multiple GPUs, please use `torchrun --nproc_per_node=4 train.py ...` instead of `python train.py ...`. We also need to use synchronized batchnorm by setting `--norm sync_batch` (or `--norm sync_instance` for instance normgalization). The `--norm batch` is not compatible with DDP.

## [Docker](docs/docker.md)

We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Datasets](docs/datasets.md)

Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)

Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)

Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset

If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)

To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request

You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/).
Please run `flake8 --ignore E501 .` and `pytest scripts/test_before_push.py -v` before you commit the code. Please also update the code structure [overview](docs/overview.md) accordingly if you add or remove files.

## Citation

If you use this code for your research, please cite our papers.

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Other Languages

[Spanish](docs/README_es.md)

## Related Projects

[img2img-turbo](https://github.com/GaParmar/img2img-turbo)<br>
[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)<br>
[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)<br>
[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)

## Cat Paper Collection

If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper [Collection](https://github.com/junyanz/CatPapers).

## Acknowledgments

Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
ssh -p 46185 root@connect.nmb2.seetacloud.com
JOGx2wic6PWn
ssh -p 43533 root@connect.nmb2.seetacloud.com
CycleGAN的效果受多个关键配置参数影响。以下是主要参数及其对模型效果的影响分析：

---

### **1. 学习率（Learning Rate）**
- **作用**：控制优化器更新模型参数的步长。
  - **过高**：可能导致训练不稳定（梯度爆炸或震荡）。
  - **过低**：收敛速度慢，可能陷入局部最优。
- **推荐设置**：
  - 初始学习率通常设为 `0.0002`（如知识库中提到的）。
  - 配合**学习率衰减策略**（如线性衰减、余弦衰减）。
- **代码示例**：
  ```python
  optimizer_G = torch.optim.Adam(
      itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
      lr=0.0002, betas=(0.5, 0.999)
  )
  scheduler_G = get_scheduler(optimizer_G, opt)  # 如线性衰减
  ```

---

### **2. 批量大小（Batch Size）**
- **作用**：
  - **小批量（如1~2）**：提高泛化能力，但训练噪声大，收敛慢。
  - **大批量**：训练更稳定，但可能过拟合或忽略数据分布细节。
- **推荐设置**：
  - 对于图像生成任务，通常选择 `1` 或 `2` 的批量大小（如知识库建议）。
  - 根据显存容量调整，避免OOM（内存不足）。

---

### **3. 损失权重（Lambda系数）**
- **循环一致性损失（`lambda_cycle`）**：
  - **作用**：控制生成图像与原始输入的匹配程度。
    - **值越大**：生成图像更接近原始输入（保留更多特征），风格迁移减弱。
    - **值越小**：风格迁移更明显，但可能丢失细节。
  - **推荐设置**：通常设为 `10.0`（如知识库示例）。
    ```python
    loss_cycle_A = criterionCycle(rec_A, real_A) * lambda_cycle_loss
    ```

- **身份映射损失（`lambda_identity`）**：
  - **作用**：当输入图像来自目标域时，要求生成器输出与输入一致（防止模式崩溃）。
    - **启用（>0）**：有助于保持颜色分布的一致性。
    - **禁用（=0）**：适用于两个域差异较大的情况。
  - **推荐设置**：通常设为 `5.0` 或 `10.0`。
    ```python
    if identity_lambda > 0:
        idt_A = netG_A2B(fake_B)
        loss_idt_A = criterionIdt(idt_A, fake_B) * beta * identity_lambda
    ```

---

### **4. 网络结构参数**
- **生成器深度**：
  - 使用更深的ResNet或U-Net结构可以捕捉更复杂的特征，但会增加计算开销。
- **判别器类型**：
  - 常用PatchGAN判别器（如70x70卷积核），输出局部真实/假概率，减少全局依赖。
- **归一化层**：
  - **InstanceNorm**：适用于单张图像归一化，避免风格泄露。
  - **SyncBatchNorm**：多GPU训练时同步BatchNorm统计量。

---

### **5. 数据预处理与增强**
- **填充方式（`--preprocess`）**：
  - 如 `scale_width`, `crop`, `resize_and_crop` 等，影响输入图像的分布。
- **数据增强**：
  - 随机翻转、旋转、裁剪等，提升模型泛化能力。
- **未配对数据集（`--dataset_mode unaligned`）**：
  - 确保两个域的数据分布差异足够大，但需避免完全不相关。

---

### **6. 优化器与训练策略**
- **优化器选择**：
  - 使用Adam优化器（`betas=(0.5, 0.999)`）是常见配置。
- **判别器更新频率（`--D_update_ratio`）**：
  - 增加判别器训练次数（如 `D_update_ratio=2`）可提升生成器的对抗能力。
- **早停机制（Early Stopping）**：
  - 监控验证集的损失或生成质量，避免过拟合。

---

### **7. 其他关键参数**
- **迭代次数（`--n_epochs`）**：
  - 典型训练周期为100~200轮，需根据数据集复杂度调整。
- **图像池大小（`--pool_size`）**：
  - 缓存历史生成图像供判别器使用（如 `pool_size=50`），避免模式崩溃。
- **随机种子（`--seed`）**：
  - 控制初始化和数据打乱的随机性，确保实验可复现。

---

### **8. 超参数调优建议**
1. **默认配置**：
   - 从论文或开源实现（如官方PyTorch CycleGAN）的默认参数开始。
2. **逐步调整**：
   - 优先调整 `lambda_cycle` 和 `lambda_identity`，观察生成图像的风格迁移程度。
3. **监控指标**：
   - 记录训练损失（`loss_G`, `loss_D`, `loss_cycle`）和生成图像的视觉效果。
4. **实验对比**：
   - 通过消融实验（Ablation Study）验证不同参数组合的影响。

---

### **总结**
| 参数类别           | 关键参数                  | 推荐范围/值               | 影响方向                     |
|--------------------|---------------------------|---------------------------|------------------------------|
| 学习率             | `lr`                      | 0.0002                    | 训练稳定性与收敛速度         |
| 批量大小           | `batch_size`              | 1~2                       | 泛化能力与训练效率           |
| 损失权重           | `lambda_cycle`            | 10.0                      | 风格迁移与细节保留的平衡     |
|                    | `lambda_identity`         | 5.0~10.0                  | 颜色一致性与模式崩溃控制     |
| 网络结构           | 生成器深度                | ResNet-9                  | 特征提取能力                 |
| 数据预处理         | `--preprocess`            | `resize_and_crop`         | 输入分布与多样性             |
| 训练策略           | `D_update_ratio`          | 1~2                       | 生成器与判别器的对抗强度     |

通过合理配置这些参数，可以显著提升CycleGAN的生成效果。建议从默认值开始，逐步调整并监控实验结果。



一、 项目整体架构与核心组件
base_model.py (抽象基类):
这是所有具体模型（如 Pix2PixModel, CycleGANModel）的父类。
它定义了模型必须实现的抽象方法：set_input, forward, optimize_parameters。这强制了所有子模型遵循统一的接口。
它提供了通用功能：
网络管理: 通过 model_names 列表跟踪需要保存/加载的网络（如 ["G", "D"]）。
损失管理: 通过 loss_names 列表跟踪需要打印/记录的损失（如 ["G_GAN", "G_L1"]）。
可视化管理: 通过 visual_names 列表跟踪需要显示/保存的图像（如 ["real_A", "fake_B", "real_B"]）。
设备与分布式: 处理 GPU 设备分配和 DDP（Distributed Data Parallel）初始化。
生命周期方法: setup (初始化、加载预训练模型、创建学习率调度器), save_networks, load_networks, update_learning_rate, get_current_visuals, get_current_losses 等。
networks.py (网络工厂):
这是一个工具库，包含了构建生成器（Generator）和判别器（Discriminator）所需的所有模块。
define_G / define_D: 根据传入的字符串参数（如 netG="unet_256", netD="basic"）动态创建对应的网络实例。
生成器 (ResnetGenerator, UnetGenerator): 实现了 ResNet 和 U-Net 架构，这是图像转换任务中最常用的两种编码器-解码器结构。
判别器 (NLayerDiscriminator, PixelDiscriminator): 实现了 PatchGAN（局部判别）和 PixelGAN（像素级判别）。
GANLoss: 一个通用的 GAN 损失计算类，支持 vanilla (原始 GAN, BCEWithLogitsLoss), lsgan (最小二乘 GAN, MSELoss) 等模式。
辅助函数: 如权重初始化 (init_weights)、学习率调度器 (get_scheduler)、归一化层 (get_norm_layer) 等。
具体模型文件 (如 pix2pix_model.py, cycle_gan_model.py):
这些是 BaseModel 的具体实现，代表了不同的图像转换算法。
Pix2PixModel: 实现了 配对数据 的图像转换。它使用条件 GAN（cGAN），其中判别器不仅看生成的图像 fake_B，还同时看输入图像 real_A（即 torch.cat((real_A, fake_B), 1)）。损失函数 = GAN Loss + λ * L1 Loss。要求数据集模式为 aligned。
CycleGANModel: 实现了 非配对数据 的图像转换。它使用两个生成器（G_A: A->B, G_B: B->A）和两个判别器（D_A, D_B）。核心是循环一致性损失（Cycle Consistency Loss），确保 G_B(G_A(A)) ≈ A 和 G_A(G_B(B)) ≈ B。可选恒等映射损失（Identity Loss）以保留颜色等信息。要求数据集模式为 unaligned。
ColorizationModel: 是 Pix2PixModel 的子类，专门用于黑白图像上色。它在 Lab 色彩空间中工作，输入是 L 通道，输出是 ab 通道。重写了可视化方法以显示 RGB 图像。
TestModel: 仅用于测试阶段，加载一个预训练的生成器并对单张图像进行推理。
TemplateModel: 一个简单的模板，展示了如何从头开始构建自己的模型，只使用 L1 回归损失。
train.py (训练主脚本):
这是程序的入口，负责协调上述所有组件完成训练流程。

