"""
YOLOv8 训练脚本

用法:
    python train_yolo.py --config configs/yolo.yaml
    python train_yolo.py --config configs/yolo.yaml --batch_size 32 --epochs 200
"""

import argparse
import os
import sys
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training")
    parser.add_argument("--config", type=str, default="configs/yolo.yaml")
    parser.add_argument("--data_yaml", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)

    set_seed(cfg.get("seed", 42))

    # 设备处理
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda":
        import torch
        if not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"

    # 输出目录
    output_dir = cfg.get("output_dir", "./runs/yolo")
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model_size = cfg.get("model_size", "n")
    pretrained = cfg.get("pretrained", True)

    if cfg.get("resume"):
        print(f"Resuming from: {cfg['resume']}")
        model = YOLO(cfg["resume"])
    else:
        model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

    print(f"Device: {device_str}")
    print(f"Output directory: {output_dir}")
    print()

    # 训练参数
    train_args = {
        "data": cfg.get("data_yaml", "coco128.yaml"),
        "epochs": cfg.get("epochs", 100),
        "batch": cfg.get("batch_size", 16),
        "imgsz": cfg.get("imgsz", 640),
        "device": device_str,
        "workers": cfg.get("num_workers", 4),
        "project": output_dir,
        "name": "train",
        "exist_ok": True,

        # 优化器
        "optimizer": cfg.get("optimizer", "SGD"),
        "lr0": cfg.get("lr", 0.01),
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg.get("weight_decay", 5e-4),

        # 学习率调度
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "warmup_momentum": cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr": cfg.get("warmup_bias_lr", 0.1),
        "lrf": cfg.get("lrf", 0.01),

        # 数据增强
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.7),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "degrees": cfg.get("degrees", 0.0),
        "translate": cfg.get("translate", 0.1),
        "scale": cfg.get("scale", 0.5),
        "shear": cfg.get("shear", 0.0),
        "perspective": cfg.get("perspective", 0.0),
        "flipud": cfg.get("flipud", 0.0),
        "fliplr": cfg.get("fliplr", 0.5),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.0),

        # 保存
        "save": True,
        "save_period": cfg.get("save_period", -1),
        "patience": cfg.get("patience", 50),

        # 日志和可视化
        "plots": True,  # 生成训练图表

        # 其他
        "seed": cfg.get("seed", 42),
        "verbose": True,
    }

    print("Training configuration:")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")
    print("=" * 70)
    print()

    # 开始训练
    results = model.train(**train_args)

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Results saved to: {output_dir}/train")
    print("=" * 70)


if __name__ == "__main__":
    main()
