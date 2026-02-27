"""
从 VOC 检测数据集创建分类数据集
裁剪每个目标框作为分类样本
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from datasets.voc_dataset import VOC_CLASSES, VOC_CLASS_TO_IDX


class VOCClassificationDataset(Dataset):
    """VOC 分类数据集：从检测框裁剪目标"""

    def __init__(self, root, year="2007", split="trainval", transform=None,
                 min_size=32, padding=0.1):
        self.root = root
        self.transform = transform
        self.min_size = min_size
        self.padding = padding

        voc_root = os.path.join(root, f"VOC{year}")
        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.ann_dir = os.path.join(voc_root, "Annotations")

        split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f.readlines()]

        # 收集所有目标框
        self.samples = []
        for img_id in image_ids:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")

            boxes, labels = self._parse_annotation(ann_path)
            for box, label in zip(boxes, labels):
                self.samples.append({
                    'image_path': img_path,
                    'bbox': box,
                    'label': label
                })

    def _parse_annotation(self, ann_path):
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
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASS_TO_IDX[name])

        return boxes, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 加载图片
        image = Image.open(sample['image_path']).convert("RGB")
        w, h = image.size

        # 裁剪目标区域（带 padding）
        xmin, ymin, xmax, ymax = sample['bbox']
        box_w = xmax - xmin
        box_h = ymax - ymin

        # 跳过太小的框
        if box_w < self.min_size or box_h < self.min_size:
            # 返回整张图片
            crop = image
        else:
            # 外扩 padding
            pad_w = box_w * self.padding
            pad_h = box_h * self.padding
            crop_xmin = max(0, xmin - pad_w)
            crop_ymin = max(0, ymin - pad_h)
            crop_xmax = min(w, xmax + pad_w)
            crop_ymax = min(h, ymax + pad_h)

            crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        label = sample['label']

        if self.transform:
            crop = self.transform(crop)

        return crop, label


def build_voc_classification_dataset(cfg, is_train=True):
    """构建 VOC 分类数据集"""
    from torchvision import transforms as T

    data_root = cfg.get("data_root", "./data/VOCdevkit")
    year = cfg.get("voc_year", "2007")
    input_size = cfg.get("input_size", 224)

    split = "trainval" if is_train else "val"

    # 数据增强
    if is_train:
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    return VOCClassificationDataset(
        root=data_root,
        year=year,
        split=split,
        transform=transform,
        min_size=cfg.get("min_crop_size", 32),
        padding=cfg.get("crop_padding", 0.1)
    )
