# Lingjing Flower Classification

灵境竞赛组后端 AI 方向第一阶段项目。项目完成了 Oxford Flowers 102
十类花卉图像分类，包含数据分析、PyTorch 训练流程、手写 ResNet18、
迁移学习、模型评估和单张图片预测。

## 主要结果

| 模型 | 测试准确率 | Macro F1 | 说明 |
|---|---:|---:|---|
| SimpleCNN | 69.75% | 0.6491 | 基础训练流程对照 |
| Manual ResNet18 | 92.17% | 0.9163 | 手写残差块，从头训练 |
| Pretrained ResNet18 | 98.93% | 0.9902 | ImageNet 预训练后微调 |

测试集共 281 张图片。迁移学习模型正确分类 278 张。

## 项目结构

```text
.
├── analysis/
│   ├── prepare_flower10.py
│   ├── extract_image_stats.py
│   └── plot_dataset_analysis.py
├── configs/
│   └── flower10_classes.json
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── outputs/
│   ├── analysis/
│   ├── baseline/
│   ├── resnet18/
│   ├── transfer/
│   ├── evaluation/
│   └── predictions/
├── src/
│   ├── data.py
│   ├── engine.py
│   ├── simple_cnn.py
│   ├── resnet.py
│   ├── train_baseline.py
│   ├── train_resnet.py
│   ├── train_transfer.py
│   ├── model_factory.py
│   ├── evaluate_models.py
│   └── predict.py
├── environment.yml
├── requirements.txt
└── README.md