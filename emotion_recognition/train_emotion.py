"""
FER2013 情绪分类训练脚本（ResNet）

用法:
    python train_emotion.py --config configs/fer2013.yaml
    python train_emotion.py --config configs/fer2013.yaml --device mps --epochs 60
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

from utils import setup_logger, AverageMeter, set_seed, format_time


EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="FER2013 Emotion Classification")
    parser.add_argument("--config", type=str, default="configs/fer2013.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
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


def build_model(cfg):
    """构建 ResNet，支持单通道灰度图输入"""
    model_name = cfg.get("model_name", "resnet18")
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", False)
    in_channels = cfg.get("in_channels", 1)

    model_fn = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }.get(model_name)
    if model_fn is None:
        raise ValueError(f"Unknown model: {model_name}")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = model_fn(weights=weights)

    # 灰度图：把 conv1 的 3 通道改为 1 通道，保留预训练权重（取均值）
    if in_channels == 1:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loss(cfg):
    loss_type = cfg.get("loss_type", "cross_entropy")
    if loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    return nn.CrossEntropyLoss()


def build_scheduler(optimizer, cfg):
    import math
    from torch.optim.lr_scheduler import LambdaLR

    warmup = cfg.get("warmup_epochs", 0)
    total = cfg["epochs"]
    min_lr = cfg.get("min_lr", 0)
    base_lr = cfg["lr"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(1e-6 / base_lr, (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, factor)

    return LambdaLR(optimizer, lr_lambda)


def build_datasets(cfg):
    """构建 FER2013 数据集，使用 ImageFolder"""
    data_root = cfg["data_root"]
    input_size = cfg.get("input_size", 48)
    in_channels = cfg.get("in_channels", 1)

    if in_channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        grayscale = transforms.Grayscale(num_output_channels=1)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        grayscale = None

    train_tfm = []
    if grayscale:
        train_tfm.append(grayscale)
    train_tfm.extend([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
    ])

    val_tfm = []
    if grayscale:
        val_tfm.append(grayscale)
    val_tfm.extend([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"),
        transform=transforms.Compose(train_tfm),
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, "test"),
        transform=transforms.Compose(val_tfm),
    )
    return train_dataset, val_dataset


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / total:.2f}%")

    return loss_meter.avg, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return loss_meter.avg, 100.0 * correct / total


def main():
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg.get("seed", 42))

    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    output_dir = cfg.get("output_dir", "./outputs/emotion_recognition")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir, name="train_emotion")

    logger.info("=" * 60)
    logger.info("FER2013 Emotion Classification")
    for k, v in sorted(cfg.items()):
        logger.info(f"  {k}: {v}")
    logger.info(f"  device (actual): {device_str}")
    logger.info("=" * 60)

    # 数据集
    logger.info("Building datasets...")
    train_dataset, val_dataset = build_datasets(cfg)
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # 模型
    model = build_model(cfg)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg['model_name']}, params: {total_params:,}")

    # 损失 / 优化器 / 调度器
    criterion = build_loss(cfg)
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    momentum=cfg.get("momentum", 0.9), weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, cfg)

    # 恢复
    start_epoch = 0
    best_acc = 0.0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # 训练
    epochs = cfg["epochs"]
    total_start = time.time()
    logger.info(f"Start training for {epochs} epochs...\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train: loss={train_loss:.4f} acc={train_acc:.2f}%  |  "
            f"Val: loss={val_loss:.4f} acc={val_acc:.2f}%  |  "
            f"LR: {current_lr:.6f}  Time: {format_time(epoch_time)}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("save_best", True):
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg,
                }, os.path.join(output_dir, "best.pt"))
                logger.info(f"  -> Best model saved (acc={best_acc:.2f}%)")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": cfg,
        }, os.path.join(output_dir, "last.pt"))

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training complete! Total time: {format_time(total_time)}")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Weights saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
