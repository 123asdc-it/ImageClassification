"""
VOC 分类训练脚本（从检测框裁剪）

用法:
    python train_voc_cls.py --config configs/voc_classification.yaml
    python train_voc_cls.py --config configs/voc_classification.yaml --batch_size 64 --epochs 50
"""

import argparse
import os
import time

import torch

# 必须在创建 DataLoader 之前调用，减少多进程对文件描述符的占用，避免 "Too many open files"
torch.multiprocessing.set_sharing_strategy("file_system")

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.model import build_model
from losses import build_loss
from optim import build_scheduler
from utils import setup_logger, AverageMeter, set_seed, format_time
from datasets.voc_classification import build_voc_classification_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="VOC Classification Training")
    parser.add_argument("--config", type=str, default="configs/voc_classification.yaml")
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


def train_one_epoch(model, loader, criterion, optimizer, device, logger, epoch):
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

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}",
                        acc=f"{100.0 * correct / total:.2f}%")

    acc = 100.0 * correct / total
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device, logger, epoch):
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

    acc = 100.0 * correct / total
    return loss_meter.avg, acc


def main():
    args = parse_args()
    cfg = load_config(args)

    set_seed(cfg.get("seed", 42))

    # 设备
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # 输出目录
    output_dir = cfg.get("output_dir", "../../../outputs/voc_classification")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir, name="train_voc_cls")

    logger.info("=" * 60)
    logger.info("VOC Classification Training Configuration:")
    for k, v in sorted(cfg.items()):
        logger.info(f"  {k}: {v}")
    logger.info(f"  device (actual): {device_str}")
    logger.info("=" * 60)

    # 数据集
    logger.info("Building datasets...")
    train_dataset = build_voc_classification_dataset(cfg, is_train=True)
    val_dataset = build_voc_classification_dataset(cfg, is_train=False)
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val   samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # 模型
    logger.info(f"Building model: {cfg['model_name']}...")
    model = build_model(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", False),
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # 损失函数
    criterion = build_loss(cfg.get("loss_type", "cross_entropy"))

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # 学习率调度器
    scheduler = build_scheduler(optimizer, cfg)

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # 训练循环
    epochs = cfg["epochs"]
    total_start = time.time()

    logger.info(f"Start training for {epochs} epochs...")
    logger.info("")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, logger, epoch
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train: loss={train_loss:.4f} acc={train_acc:.2f}%  |  "
            f"Val: loss={val_loss:.4f} acc={val_acc:.2f}%  |  "
            f"LR: {current_lr:.6f}  Time: {format_time(epoch_time)}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("save_best", True):
                save_path = os.path.join(output_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg,
                }, save_path)
                logger.info(f"  -> Best model saved (acc={best_acc:.2f}%)")

        # 保存最新 checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": cfg,
        }, os.path.join(output_dir, "last.pt"))

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training complete! Total time: {format_time(total_time)}")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Weights saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
