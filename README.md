# PyTorch 计算机视觉项目集

基于 PyTorch 的计算机视觉训练框架，包含三个独立任务项目。

## 项目结构

```
ImageClassification/
├── voc_two_stage/           # 项目1: VOC 两阶段检测+分类
│   ├── configs/             # 配置文件
│   ├── datasets/            # 数据加载模块
│   ├── losses/              # 损失函数
│   ├── models/              # 模型定义 (SSD, ResNet)
│   ├── optim/               # 优化器和调度器
│   ├── utils/               # 工具函数
│   ├── data/                # VOC2007 数据集
│   ├── weights/             # 训练权重
│   ├── outputs/             # 测试输出
│   ├── train_det.py         # SSD 检测训练
│   ├── train_voc_cls.py     # VOC 分类训练
│   ├── eval_voc_cls.py      # 分类评估
│   ├── test_det.py          # 检测测试
│   ├── test_two_stage_voc.py # 两阶段测试
│   └── pipeline.py          # 检测+分类流水线
│
├── emotion_recognition/     # 项目2: 情绪识别
│   ├── configs/             # 配置文件
│   ├── models/              # 模型定义 (ResNet, 复制自 VOC 项目)
│   ├── utils/               # 工具函数 (复制自 VOC 项目)
│   ├── data/                # FER-2013 数据集
│   ├── weights/             # 训练权重
│   ├── outputs/             # 测试输出
│   ├── train_emotion.py     # 情绪分类训练
│   └── eval_emotion.py      # 情绪分类评估
│
├── yolo_traffic/            # 项目3: YOLO 交通物体检测
│   ├── configs/             # 配置文件
│   ├── scripts/             # 脚本
│   │   ├── train_yolo.py    # YOLO 训练
│   │   ├── test_yolo.py     # YOLO 测试
│   │   ├── visualize.py     # 结果可视化
│   │   └── download_coco_traffic.py  # 数据集下载
│   ├── utils/               # 工具函数
│   ├── data/                # COCO 交通子集数据
│   ├── weights/             # 训练权重 + 预训练模型
│   └── outputs/             # 训练和测试输出
│
├── docs/                    # 文档和报告
├── requirements.txt         # 依赖列表
└── README.md                # 本文件
```

---

## 项目1: VOC 两阶段检测+分类

SSD 目标检测 + ResNet 图像分类，基于 VOC2007 数据集。

```bash
cd voc_two_stage

# 训练检测器
python train_det.py --config configs/detection.yaml

# 训练分类器
python train_voc_cls.py --config configs/voc_classification.yaml

# 两阶段测试
python test_two_stage_voc.py --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg --device mps
```

## 项目2: 情绪识别

ResNet18 情绪分类，基于 FER-2013 数据集（7 类表情）。

```bash
cd emotion_recognition

# 训练
python train_emotion.py --config configs/fer2013.yaml --device mps

# 评估
python eval_emotion.py --weights weights/best.pt --device mps
```

## 项目3: YOLO 交通物体检测

YOLOv8 检测行人、自行车、汽车、摩托车、公交车、卡车。

```bash
cd yolo_traffic

# 训练
python3 -c "
from ultralytics import YOLO
model = YOLO('weights/yolov8n.pt')
model.train(
    data='data/data.yaml',
    epochs=100, imgsz=640, device='cpu',
    batch=8, workers=0, mosaic=0,
    project='weights/coco_traffic', name='train', exist_ok=True
)
"

# 测试
python scripts/test_yolo.py --weights weights/coco_traffic/train/weights/best.pt --source path/to/image.jpg --save
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

## 技术栈

- **框架**: PyTorch, torchvision, ultralytics (YOLOv8)
- **模型**: ResNet18, SSD, YOLOv8
- **数据集**: VOC2007, FER-2013, COCO 交通子集
