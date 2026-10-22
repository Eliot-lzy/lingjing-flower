# 项目运行视频录制脚本

建议视频长度为 3 到 5 分钟。录制时重点展示真实运行结果、项目结构和
手写 ResNet 代码，不需要播放完整训练过程。

## 录制前准备

1. 打开项目目录并激活环境：

```powershell
Set-Location 'D:\PythonProject\lingjing-flower'
conda activate eliotpytorch
```

2. 检查项目状态：

```powershell
git status
git log --oneline --graph --decorate -10
```

3. 提前打开输出目录，准备好以下文件：

```text
outputs/analysis/class_distribution.png
outputs/analysis/image_dimensions.png
outputs/analysis/brightness_by_class.png
outputs/analysis/class_samples.png
outputs/resnet18/training_curves.png
outputs/transfer/training_curves.png
outputs/evaluation/model_comparison.png
outputs/evaluation/confusion_matrix_pretrained_resnet18.png
outputs/predictions/image_00001_prediction.png
```

## 00:00 - 00:20 项目介绍

展示 README 和项目目录，介绍：

- 项目名称：灵境花卉十类图像分类
- 数据集：Oxford Flowers 102 十类子集
- 数据规模：1873 张图片
- 完成内容：数据分析、SimpleCNN、手写 ResNet18、迁移学习、模型评估和单张图片推理

## 00:20 - 00:45 开发历史

在终端运行：

```powershell
git log --oneline --graph --decorate -10
```

说明项目按照数据准备、数据分析、数据加载、模型实现、模型训练和模型
评估分阶段提交，保留了真实开发历史。

## 00:45 - 01:10 环境与 GPU

运行：

```powershell
python src\check_environment.py
```

展示：

- Python 版本
- PyTorch 版本
- CUDA 可用
- RTX 5060 Laptop GPU
- GPU smoke test OK

## 01:10 - 01:40 数据分析结果

打开并依次展示：

```text
outputs/analysis/class_distribution.png
outputs/analysis/image_dimensions.png
outputs/analysis/brightness_by_class.png
outputs/analysis/class_samples.png
```

说明项目使用 NumPy、Pandas 和 Matplotlib 完成了类别数量、图片尺寸和
亮度分布分析，并展示了每个类别的样本图片。

## 01:40 - 02:20 手写 ResNet18

打开 `src/resnet.py`，重点展示：

- `BasicBlock`
- `conv1` 和 `conv2`
- `shortcut`
- Identity 和 1×1 卷积的维度匹配
- 最后的残差相加

运行维度检查：

```powershell
python -c "import torch; from src.resnet import BasicBlock, ResNet18; model=ResNet18(); y=model(torch.randn(2,3,128,128)); print('output:', tuple(y.shape)); print('params:', sum(p.numel() for p in model.parameters())); b1=BasicBlock(64,64); b2=BasicBlock(64,128,stride=2); print('same:', type(b1.shortcut).__name__); print('match:', type(b2.shortcut[0]).__name__, b2.shortcut[0].stride)"
```

说明这个 ResNet18 没有调用 `torchvision.models.resnet18`，而是自行
实现了残差块、下采样和维度匹配，并手动组装四个 stage。

## 02:20 - 02:50 训练结果

打开：

```text
outputs/resnet18/training_curves.png
outputs/transfer/training_curves.png
```

说明：

- 手写 ResNet18 测试准确率：92.17%
- 迁移学习 ResNet18 测试准确率：98.93%
- 两个模型均在 NVIDIA RTX 5060 Laptop GPU 上实际训练完成

可以补充说明手写 ResNet18 训练 20 轮，耗时约 1.6 分钟；迁移学习采用
先训练分类头、再全量微调的两阶段策略。

## 02:50 - 03:20 模型评估

打开：

```text
outputs/evaluation/model_comparison.png
outputs/evaluation/confusion_matrix_pretrained_resnet18.png
```

说明三个模型的测试结果：

- SimpleCNN：69.75%
- 手写 ResNet18：92.17%
- 迁移学习 ResNet18：98.93%

指出测试集共 281 张图片，迁移学习模型正确分类 278 张。还可以说明
SimpleCNN 对 rose 和 water lily 存在明显混淆，而预训练特征显著
改善了分类效果。

## 03:20 - 04:00 单张图片预测

运行：

```powershell
python -m src.predict data\raw\flowers-102\jpg\image_00001.jpg --top-k 3
```

展示终端结果，并打开：

```text
outputs/predictions/image_00001_prediction.png
```

说明该图片的真实类别为 passion flower，模型 Top-1 预测也是
passion flower，同时展示 Top-2 和 Top-3 的概率。

再演示手写 ResNet18 模型的推理能力：

```powershell
python -m src.predict data\raw\flowers-102\jpg\image_00948.jpg --checkpoint outputs\models\resnet18_manual_best.pt --top-k 3
```

打开生成的预测图片，说明同一个预测程序可以通过 checkpoint 自动加载
不同模型结构。

## 04:00 - 04:20 总结

总结项目完成内容：

- 完成了从数据分析到模型推理的端到端图像分类流程
- 实际使用了 Python、NumPy、Pandas、Matplotlib 和 PyTorch
- 手动实现残差块并组装 ResNet18
- 使用迁移学习完成性能对照
- 通过 Git 保留了分阶段开发历史
- 能够对单张新图片输出 Top-K 分类结果和置信度

结束语可以概括为：

> 本项目让我理解了图像分类不仅是训练一个网络，还需要合理划分数据、
> 分析数据分布、保证预处理一致、保存最佳模型，并通过评估和推理验证
> 最终结果是否真正可用。