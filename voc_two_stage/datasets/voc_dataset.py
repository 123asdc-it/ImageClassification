"""
Pascal VOC 目标检测数据集加载。

支持 VOC2007 / VOC2012 格式：
  VOCdevkit/
    VOC2007/
      JPEGImages/    *.jpg
      Annotations/   *.xml
      ImageSets/Main/  train.txt, val.txt, trainval.txt, test.txt
"""

import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image

# VOC 20 类 + 背景
VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}


class VOCDetectionDataset(Dataset):
    """
    Pascal VOC 目标检测数据集。
    返回: image (PIL), targets dict {boxes: (N,4), labels: (N,)}
    boxes 格式: [xmin, ymin, xmax, ymax]，归一化到 [0, 1]
    """

    def __init__(self, root, year="2007", split="trainval", transform=None):
        self.root = root
        self.transform = transform

        voc_root = os.path.join(root, f"VOC{year}")
        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.ann_dir = os.path.join(voc_root, "Annotations")

        split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]

        # 加载图片
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # 解析标注
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        boxes, labels = self._parse_annotation(ann_path, w, h)

        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.transform is not None:
            image, targets = self.transform(image, targets)

        return image, targets

    def _parse_annotation(self, ann_path, img_w, img_h):
        """解析 VOC XML 标注文件"""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue

            name = obj.find("name").text.strip()
            if name not in VOC_CLASS_TO_IDX:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / img_w
            ymin = float(bbox.find("ymin").text) / img_h
            xmax = float(bbox.find("xmax").text) / img_w
            ymax = float(bbox.find("ymax").text) / img_h

            # 确保坐标合法
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(0.0, min(1.0, xmax))
            ymax = max(0.0, min(1.0, ymax))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(VOC_CLASS_TO_IDX[name])

        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
            labels = [0]

        return boxes, labels


def build_voc_dataset(cfg, is_train=True):
    """根据配置构建 VOC 数据集"""
    from datasets.det_transforms import DetectionTransform

    data_root = cfg.get("det_data_root", "./data/VOCdevkit")
    year = cfg.get("voc_year", "2007")
    input_size = cfg.get("det_input_size", 300)

    split = "trainval" if is_train else "val"
    transform = DetectionTransform(input_size=input_size, is_train=is_train)

    return VOCDetectionDataset(
        root=data_root,
        year=year,
        split=split,
        transform=transform,
    )
