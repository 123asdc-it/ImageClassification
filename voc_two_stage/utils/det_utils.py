"""
目标检测工具函数：
- IoU 计算
- Anchor 与 GT 匹配
- BBox 编码/解码
- NMS 后处理
"""

import torch


def box_iou(boxes1, boxes2):
    """
    计算两组 boxes 之间的 IoU。
    Args:
        boxes1: (N, 4) [xmin, ymin, xmax, ymax]
        boxes2: (M, 4) [xmin, ymin, xmax, ymax]
    Returns:
        iou: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_xmin = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_ymin = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_xmax = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_ymax = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * \
                 (inter_ymax - inter_ymin).clamp(min=0)

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def encode_boxes(matched_gt_boxes, anchors):
    """
    将 GT boxes 编码为相对于 anchor 的偏移量。
    Args:
        matched_gt_boxes: (N, 4) [xmin, ymin, xmax, ymax]
        anchors: (N, 4) [cx, cy, w, h]
    Returns:
        targets: (N, 4) [tx, ty, tw, th]
    """
    # GT: xyxy -> cxcywh
    gt_cx = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
    gt_cy = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
    gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]

    tx = (gt_cx - anchors[:, 0]) / anchors[:, 2].clamp(min=1e-6)
    ty = (gt_cy - anchors[:, 1]) / anchors[:, 3].clamp(min=1e-6)
    tw = torch.log(gt_w / anchors[:, 2].clamp(min=1e-6) + 1e-6)
    th = torch.log(gt_h / anchors[:, 3].clamp(min=1e-6) + 1e-6)

    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(reg_preds, anchors):
    """
    将预测的偏移量解码为实际 bbox 坐标。
    Args:
        reg_preds: (N, 4) [tx, ty, tw, th]
        anchors: (N, 4) [cx, cy, w, h]
    Returns:
        boxes: (N, 4) [xmin, ymin, xmax, ymax]
    """
    cx = reg_preds[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = reg_preds[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(reg_preds[:, 2].clamp(max=10)) * anchors[:, 2]
    h = torch.exp(reg_preds[:, 3].clamp(max=10)) * anchors[:, 3]

    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def match_anchors_to_gt(anchors, gt_boxes, gt_labels,
                        pos_threshold=0.5, neg_threshold=0.3):
    """
    将 anchor 与 GT boxes 匹配。
    Args:
        anchors: (A, 4) [cx, cy, w, h] 格式
        gt_boxes: (G, 4) [xmin, ymin, xmax, ymax] 格式
        gt_labels: (G,) 类别标签
        pos_threshold: IoU >= 此值为正样本
        neg_threshold: IoU < 此值为负样本
    Returns:
        cls_targets: (A,) 每个 anchor 的类别（0=背景）
        reg_targets: (A, 4) 编码后的偏移量
    """
    device = anchors.device
    num_anchors = anchors.size(0)

    # anchor cxcywh -> xyxy 用于计算 IoU
    anchor_xyxy = torch.stack([
        anchors[:, 0] - anchors[:, 2] / 2,
        anchors[:, 1] - anchors[:, 3] / 2,
        anchors[:, 0] + anchors[:, 2] / 2,
        anchors[:, 1] + anchors[:, 3] / 2,
    ], dim=1)

    if gt_boxes.numel() == 0 or (gt_boxes.sum() == 0):
        cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
        reg_targets = torch.zeros(num_anchors, 4, dtype=torch.float32, device=device)
        return cls_targets, reg_targets

    iou = box_iou(anchor_xyxy, gt_boxes)  # (A, G)
    max_iou, max_idx = iou.max(dim=1)     # 每个 anchor 最匹配的 GT

    # 默认全部为背景
    cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
    reg_targets = torch.zeros(num_anchors, 4, dtype=torch.float32, device=device)

    # 正样本
    pos_mask = max_iou >= pos_threshold
    cls_targets[pos_mask] = gt_labels[max_idx[pos_mask]]

    # 确保每个 GT 至少有一个匹配的 anchor
    best_anchor_per_gt = iou.argmax(dim=0)  # (G,)
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
        cls_targets[anchor_idx] = gt_labels[gt_idx]
        pos_mask[anchor_idx] = True

    # 编码正样本的 bbox
    matched_gt = gt_boxes[max_idx]
    reg_targets[pos_mask] = encode_boxes(
        matched_gt[pos_mask], anchors[pos_mask]
    )

    # 忽略 IoU 在 neg_threshold 和 pos_threshold 之间的 anchor
    # （设为 -1，在 loss 中忽略）
    ignore_mask = (max_iou >= neg_threshold) & (~pos_mask)
    cls_targets[ignore_mask] = -1

    return cls_targets, reg_targets


def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制 (NMS)。
    Args:
        boxes: (N, 4) [xmin, ymin, xmax, ymax]
        scores: (N,)
        iou_threshold: IoU 阈值
    Returns:
        keep: 保留的索引
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    # 按分数降序排列
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        remaining = order[1:]
        iou_vals = box_iou(
            boxes[i].unsqueeze(0), boxes[remaining]
        ).squeeze(0)

        mask = iou_vals <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
