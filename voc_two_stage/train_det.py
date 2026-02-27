"""
目标检测训练脚本 (SSD + VOC)

用法:
    python train_det.py --config configs/detection.yaml
    python train_det.py --config configs/detection.yaml --batch_size 8 --epochs 200
"""

import argparse
import os
import time

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.detector import build_detector, AnchorGenerator
from losses.detection_loss import build_detection_loss
from datasets.voc_dataset import build_voc_dataset
from datasets.det_transforms import detection_collate_fn
from optim import build_scheduler
from utils import setup_logger, AverageMeter, set_seed, format_time
from utils.det_utils import match_anchors_to_gt


# ======================== 配置 ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Training (SSD)")
    parser.add_argument("--config", type=str, default="configs/detection.yaml")
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


# ======================== 训练 & 验证 ========================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    anchors, cfg, logger, epoch):
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pos_thresh = cfg.get("pos_threshold", 0.5)
    neg_thresh = cfg.get("neg_threshold", 0.3)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        B = images.size(0)

        # 前向传播
        cls_preds, reg_preds = model(images)

        # 为每张图构建训练目标
        all_cls_targets = []
        all_reg_targets = []
        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)

            cls_t, reg_t = match_anchors_to_gt(
                anchors, gt_boxes, gt_labels,
                pos_threshold=pos_thresh,
                neg_threshold=neg_thresh,
            )
            all_cls_targets.append(cls_t)
            all_reg_targets.append(reg_t)

        cls_targets = torch.stack(all_cls_targets)  # (B, A)
        reg_targets = torch.stack(all_reg_targets)  # (B, A, 4)

        # 计算损失（loss 内部处理 -1 忽略标签）
        total_loss, cls_loss, reg_loss = criterion(
            cls_preds, reg_preds, cls_targets, reg_targets
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_meter.update(total_loss.item(), B)
        cls_loss_meter.update(cls_loss.item(), B)
        reg_loss_meter.update(reg_loss.item(), B)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            cls=f"{cls_loss_meter.avg:.4f}",
            reg=f"{reg_loss_meter.avg:.4f}",
        )

    return loss_meter.avg, cls_loss_meter.avg, reg_loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, anchors, cfg, logger, epoch):
    model.eval()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pos_thresh = cfg.get("pos_threshold", 0.5)
    neg_thresh = cfg.get("neg_threshold", 0.3)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        B = images.size(0)

        cls_preds, reg_preds = model(images)

        all_cls_targets = []
        all_reg_targets = []
        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            cls_t, reg_t = match_anchors_to_gt(
                anchors, gt_boxes, gt_labels,
                pos_threshold=pos_thresh,
                neg_threshold=neg_thresh,
            )
            all_cls_targets.append(cls_t)
            all_reg_targets.append(reg_t)

        cls_targets = torch.stack(all_cls_targets)
        reg_targets = torch.stack(all_reg_targets)

        total_loss, cls_loss, reg_loss = criterion(
            cls_preds, reg_preds, cls_targets, reg_targets
        )

        loss_meter.update(total_loss.item(), B)
        cls_loss_meter.update(cls_loss.item(), B)
        reg_loss_meter.update(reg_loss.item(), B)

    return loss_meter.avg, cls_loss_meter.avg, reg_loss_meter.avg


# ======================== 主函数 ========================

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

    # 输出目录 & 日志
    output_dir = cfg.get("output_dir", "../../../outputs/detection")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir, name="train_det")

    logger.info("=" * 60)
    logger.info("Detection Training Configuration:")
    for k, v in sorted(cfg.items()):
        logger.info(f"  {k}: {v}")
    logger.info(f"  device (actual): {device_str}")
    logger.info("=" * 60)

    # 数据集
    logger.info("Building datasets...")
    train_dataset = build_voc_dataset(cfg, is_train=True)
    val_dataset = build_voc_dataset(cfg, is_train=False)
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val   samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=detection_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=detection_collate_fn,
    )

    # 模型
    num_classes = cfg["num_classes"]
    logger.info(f"Building SSD detector (num_classes={num_classes})...")
    model = build_detector(
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", False),
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # Anchor 生成
    input_size = cfg.get("det_input_size", 300)
    anchor_gen = AnchorGenerator(image_size=input_size)
    anchors = anchor_gen.generate(device=device)
    logger.info(f"  Total anchors: {anchors.size(0)}")

    # 损失函数
    criterion = build_detection_loss(
        num_classes=num_classes,
        neg_pos_ratio=cfg.get("neg_pos_ratio", 3),
    )

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 5e-4),
    )

    # 学习率调度器
    scheduler = build_scheduler(optimizer, cfg)

    # 恢复训练
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ==================== 训练循环 ====================
    epochs = cfg["epochs"]
    patience = cfg.get("patience", 20)  # 默认20个epoch的耐心值
    total_start = time.time()
    logger.info(f"Start training for {epochs} epochs...")
    logger.info(f"Early stopping patience: {patience} epochs")
    logger.info("")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_loss, train_cls, train_reg = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            anchors, cfg, logger, epoch
        )

        val_loss, val_cls, val_reg = validate(
            model, val_loader, criterion, device,
            anchors, cfg, logger, epoch
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train: {train_loss:.4f} (cls={train_cls:.4f} reg={train_reg:.4f})  |  "
            f"Val: {val_loss:.4f} (cls={val_cls:.4f} reg={val_reg:.4f})  |  "
            f"LR: {current_lr:.6f}  Time: {format_time(epoch_time)}"
        )

        # 保存最佳模型（检测用 val_loss 作为指标）
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # 重置耐心计数器
            if cfg.get("save_best", True):
                save_path = os.path.join(output_dir, "best_det.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                    "config": cfg,
                }, save_path)
                logger.info(f"  -> Best model saved (loss={best_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement for {patience_counter} epoch(s)")

            # 早停检查
            if patience_counter >= patience:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                logger.info(f"Best validation loss: {best_loss:.4f}")
                logger.info("=" * 60)
                break

        # 保存最新 checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "config": cfg,
        }, os.path.join(output_dir, "last_det.pt"))

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training complete! Total time: {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Weights saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
