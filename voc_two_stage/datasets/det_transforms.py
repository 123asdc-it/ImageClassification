"""
目标检测专用的数据增强 / 预处理。
需要同时变换图片和 bounding boxes。
"""

import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class DetectionTransform:
    """
    检测任务的数据变换：
    - Resize 到固定尺寸
    - 随机水平翻转（训练时）
    - 颜色抖动（训练时）
    - ToTensor + Normalize
    """

    def __init__(self, input_size=300, is_train=True):
        self.input_size = input_size
        self.is_train = is_train
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image, targets):
        # Resize
        image = F.resize(image, (self.input_size, self.input_size))

        if self.is_train:
            # 随机水平翻转
            if random.random() < 0.5:
                image = F.hflip(image)
                boxes = targets["boxes"]
                # 翻转 x 坐标: new_xmin = 1 - xmax, new_xmax = 1 - xmin
                boxes = boxes.clone()
                xmin = 1.0 - boxes[:, 2]
                xmax = 1.0 - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                targets["boxes"] = boxes

            # 颜色抖动
            image = T.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            )(image)

        # ToTensor + Normalize
        image = F.to_tensor(image)
        image = self.normalize(image)

        return image, targets


def detection_collate_fn(batch):
    """
    检测任务的 collate 函数。
    因为每张图的目标数量不同，不能直接 stack targets。

    Returns:
        images: (B, 3, H, W)
        targets: list of dicts, 每个 dict 包含 boxes 和 labels
    """
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets
