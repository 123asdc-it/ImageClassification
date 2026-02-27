"""
目标检测损失函数：分类损失 + 回归损失

- 分类：Focal Loss（解决正负样本不平衡）
- 回归：Smooth L1 Loss（对 bbox 偏移量）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: 降低易分类样本的权重，聚焦难分类样本。
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 分类 logits
            targets: (N,) 类别标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)  # 预测正确的概率
        focal_weight = self.alpha * (1 - p) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """
    目标检测总损失 = 分类损失 + lambda * 回归损失

    使用 hard negative mining：
    - 正样本：与 GT 的 IoU >= pos_threshold 的 anchor
    - 负样本：按分类 loss 排序取 top-k（正负比 1:3）
    """

    def __init__(self, num_classes=21, neg_pos_ratio=3, reg_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.reg_weight = reg_weight

    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets):
        """
        Args:
            cls_preds:   (B, num_anchors, num_classes)
            reg_preds:   (B, num_anchors, 4)
            cls_targets: (B, num_anchors) 类别标签，0=背景
            reg_targets: (B, num_anchors, 4) 编码后的 bbox 偏移
        Returns:
            total_loss, cls_loss, reg_loss
        """
        B = cls_preds.size(0)
        num_anchors = cls_preds.size(1)

        # ---- 正样本 mask ----
        pos_mask = cls_targets > 0  # (B, num_anchors)
        num_pos = pos_mask.sum().clamp(min=1).float()

        # ---- 回归损失（只算正样本） ----
        reg_loss = F.smooth_l1_loss(
            reg_preds[pos_mask],
            reg_targets[pos_mask],
            reduction="sum",
        ) / num_pos

        # ---- 分类损失（hard negative mining） ----
        # 将忽略标签 (-1) 临时设为 0，避免 cross_entropy 越界
        ignore_mask = cls_targets < 0
        safe_targets = cls_targets.clone()
        safe_targets[ignore_mask] = 0

        cls_preds_flat = cls_preds.view(-1, self.num_classes)
        cls_targets_flat = safe_targets.view(-1)
        all_cls_loss = F.cross_entropy(
            cls_preds_flat, cls_targets_flat, reduction="none"
        ).view(B, num_anchors)

        # 忽略的 anchor 不参与 loss
        all_cls_loss[ignore_mask] = 0

        # 正样本的分类 loss
        pos_cls_loss = all_cls_loss[pos_mask].sum()

        # 负样本：排除正样本和忽略样本，按 loss 降序取 top-k
        all_cls_loss[pos_mask] = 0  # 正样本不参与负样本排序
        _, neg_indices = all_cls_loss.sort(dim=1, descending=True)

        num_pos_per_img = pos_mask.sum(dim=1, keepdim=True)  # (B, 1)
        num_neg = (num_pos_per_img * self.neg_pos_ratio).clamp(max=num_anchors)

        # 构建负样本 mask
        rank = torch.arange(num_anchors, device=cls_preds.device).unsqueeze(0)
        neg_mask = rank < num_neg  # (B, num_anchors)

        # 按排序后的索引取负样本 loss
        sorted_loss = all_cls_loss.gather(1, neg_indices)
        neg_cls_loss = sorted_loss[neg_mask].sum()

        cls_loss = (pos_cls_loss + neg_cls_loss) / num_pos

        total_loss = cls_loss + self.reg_weight * reg_loss
        return total_loss, cls_loss, reg_loss


def build_detection_loss(num_classes=21, **kwargs):
    """构建检测损失函数"""
    return DetectionLoss(num_classes=num_classes, **kwargs)
