"""
VOC 分类评估：加载训练好的权重，在验证集上计算准确率与 loss。

用法:
    python eval_voc_cls.py --weights path/to/best.pt
    python eval_voc_cls.py --weights path/to/best.pt --device mps
    python eval_voc_cls.py --weights path/to/best.pt --config configs/voc_classification.yaml
"""

import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from models.model import build_model
from losses import build_loss
from utils import AverageMeter
from datasets.voc_classification import build_voc_classification_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="VOC Classification Evaluation")
    parser.add_argument("--weights", type=str, required=True, help="checkpoint 路径，如 best.pt / last.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="可选，覆盖 checkpoint 中的 config（如 data_root）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / total
    return loss_meter.avg, acc


def main():
    args = parse_args()
    device_str = args.device or "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # 从 checkpoint 读取 config
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    if args.config:
        with open(args.config, "r") as f:
            overlay = yaml.safe_load(f)
        for k, v in overlay.items():
            cfg[k] = v
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device

    # 模型
    model = build_model(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # 验证集
    val_dataset = build_voc_classification_dataset(cfg, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    criterion = build_loss(cfg.get("loss_type", "cross_entropy"))
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Weights: {args.weights}")
    print(f"Epoch (in ckpt): {ckpt.get('epoch', '?')}, Best acc (in ckpt): {ckpt.get('best_acc', '?')}%")
    print(f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
