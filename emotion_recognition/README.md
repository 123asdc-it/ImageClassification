# 情绪识别 (FER-2013)

基于 ResNet18 的情绪分类，使用 FER-2013 数据集（7 类表情）。

## 目录结构

```
emotion_recognition/
├── configs/
│   └── fer2013.yaml        # 训练配置
├── models/                 # 模型定义 (ResNet18, 复制自 VOC 项目)
│   └── model.py
├── utils/                  # 工具函数 (复制自 VOC 项目)
│   ├── logger.py
│   ├── meter.py
│   └── misc.py
├── data/                   # FER-2013 数据集
│   └── FER-2013/
│       ├── train/          # 训练集 (angry, disgust, fear, happy, neutral, sad, surprise)
│       └── test/           # 测试集
├── weights/                # 训练权重
│   ├── best.pt
│   └── last.pt
├── outputs/                # 输出
├── train_emotion.py        # 训练脚本
└── eval_emotion.py         # 评估/推理脚本
```

## 使用方法

```bash
cd emotion_recognition

# 训练
python train_emotion.py --config configs/fer2013.yaml --device mps

# 评估
python eval_emotion.py --weights weights/best.pt --device mps

# 单张图片推理
python eval_emotion.py --weights weights/best.pt --image path/to/face.jpg --device mps
```

## 表情类别

angry, disgust, fear, happy, neutral, sad, surprise（共 7 类）
