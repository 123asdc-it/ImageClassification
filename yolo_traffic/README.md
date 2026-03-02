# YOLO 交通物体检测

基于 YOLOv8 的交通物体检测，使用 COCO 2017 交通子集。

## 目录结构

```
yolo_traffic/
├── configs/
│   └── yolo.yaml               # YOLO 训练配置
├── scripts/
│   ├── train_yolo.py           # 训练脚本
│   ├── test_yolo.py            # 测试脚本
│   ├── visualize.py            # 结果可视化
│   └── download_coco_traffic.py # 数据集下载
├── utils/
│   ├── __init__.py             # set_seed 等工具
│   └── logger.py               # 训练可视化工具
├── data/                       # COCO 交通子集
│   ├── data.yaml               # 数据集配置
│   ├── images/                 # 图片
│   └── labels/                 # YOLO 格式标签
├── runs/                       # 训练和测试输出
│   └── detect/
│       └── coco_traffic/
│           └── train/
│               └── weights/    # 训练权重
└── README.md
```

## 检测类别

person, bicycle, car, motorcycle, bus, truck（共 6 类）

## 使用方法

```bash
# 训练（使用配置文件）
python3 scripts/train_yolo.py --config configs/yolo.yaml

# 或使用命令行参数覆盖配置
python3 scripts/train_yolo.py \
    --config configs/yolo.yaml \
    --epochs 50 \
    --batch_size 8 \
    --device cpu

# 测试
python3 scripts/test_yolo.py \
    --weights runs/detect/coco_traffic/train/weights/best.pt \
    --source data/images/val/000000174482.jpg \
    --save --device cpu

# 可视化训练结果
python3 scripts/visualize.py --results_dir runs/detect/coco_traffic/train
```
