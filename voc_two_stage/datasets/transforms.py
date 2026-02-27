from torchvision import transforms


def get_train_transforms(input_size=32):
    """训练集数据增强 pipeline"""
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_val_transforms(input_size=32):
    """验证/测试集预处理 pipeline"""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


# ---------- Mosaic 数据增强（可选，用于分类场景的简化版） ----------
import random
import numpy as np
from PIL import Image


class MosaicTransform:
    """
    Mosaic 数据增强：将 4 张图拼成一张。
    需要传入整个 dataset 引用，以便随机采样其他图片。
    在 Dataset.__getitem__ 中使用。
    """

    def __init__(self, dataset, input_size=32):
        self.dataset = dataset
        self.input_size = input_size

    def __call__(self, index):
        s = self.input_size
        # 拼接中心点
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)

        indices = [index] + random.choices(range(len(self.dataset)), k=3)
        mosaic_img = Image.new("RGB", (s, s))

        for i, idx in enumerate(indices):
            img, label = self.dataset.get_raw(idx)
            img = img.resize((s, s))

            if i == 0:    # 左上
                mosaic_img.paste(img.crop((s - xc, s - yc, s, s)), (0, 0))
            elif i == 1:  # 右上
                mosaic_img.paste(img.crop((0, s - yc, xc, s)), (xc, 0))
            elif i == 2:  # 左下
                mosaic_img.paste(img.crop((s - xc, 0, s, yc)), (0, yc))
            elif i == 3:  # 右下
                mosaic_img.paste(img.crop((0, 0, xc, yc)), (xc, yc))

        # 返回主图片的 label
        _, main_label = self.dataset.get_raw(index)
        return mosaic_img, main_label
