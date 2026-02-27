# VOC 两阶段检测+分类

SSD 目标检测 + ResNet 图像分类，基于 VOC2007 数据集。

## 目录结构

```
voc_two_stage/
├── configs/                   # 配置文件
│   ├── detection.yaml         # SSD 检测训练配置
│   ├── voc_classification.yaml # VOC 分类训练配置
│   └── pipeline.yaml          # 两阶段流水线配置
├── datasets/                  # 数据加载模块
│   ├── voc_dataset.py         # VOC 检测数据集
│   ├── voc_classification.py  # VOC 分类数据集
│   ├── det_transforms.py      # 检测数据增强
│   └── transforms.py          # 分类数据增强
├── losses/                    # 损失函数
│   ├── loss.py                # 分类损失 (CrossEntropy)
│   └── detection_loss.py      # 检测损失 (Focal Loss + Smooth L1)
├── models/                    # 模型定义
│   ├── model.py               # 分类模型 (ResNet18)
│   └── detector.py            # 检测模型 (SSD)
├── optim/                     # 优化器
│   └── scheduler.py           # 学习率调度 (Warmup + Cosine)
├── utils/                     # 工具函数
│   ├── logger.py              # 日志系统
│   ├── meter.py               # 指标统计
│   ├── det_utils.py           # 检测工具 (IoU, NMS, Anchor)
│   └── misc.py                # 通用工具
├── data/                      # VOC2007 数据集
│   └── VOCdevkit/VOC2007/
├── weights/                   # 训练权重
│   ├── detection/             # SSD 检测权重
│   └── voc_classification/    # 分类权重
├── outputs/                   # 测试输出
├── train_det.py               # 检测训练脚本
├── train_voc_cls.py           # 分类训练脚本
├── eval_voc_cls.py            # 分类评估
├── test_det.py                # 检测测试
├── test_two_stage_voc.py      # 两阶段测试
└── pipeline.py                # 检测+分类流水线
```

## 使用方法

```bash
cd voc_two_stage

# 训练 SSD 检测器
python train_det.py --config configs/detection.yaml

# 训练 VOC 分类器
python train_voc_cls.py --config configs/voc_classification.yaml

# 两阶段测试：检测 → 分类 → 可视化
python test_two_stage_voc.py \
    --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
    --device mps

# 检测测试
python test_det.py --weights weights/detection/best_det.pt --image path/to/image.jpg

# 分类评估
python eval_voc_cls.py --weights weights/voc_classification/best.pt --device mps
```
