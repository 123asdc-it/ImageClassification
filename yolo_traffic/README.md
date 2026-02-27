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
├── weights/
│   ├── yolov8n.pt              # 预训练 YOLOv8n
│   ├── yolov8s.pt              # 预训练 YOLOv8s
│   └── coco_traffic/train/weights/  # 训练权重
├── outputs/                    # 训练和测试输出
└── README.md
```

## 检测类别

person, bicycle, car, motorcycle, bus, truck（共 6 类）

## 使用方法

```bash
cd yolo_traffic

# 训练 (使用 Python API 避免 mosaic shape mismatch bug)
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
python scripts/test_yolo.py \
    --weights weights/coco_traffic/train/weights/best.pt \
    --source path/to/image.jpg --save

# 可视化训练结果
python scripts/visualize.py --results_dir weights/coco_traffic/train
```
