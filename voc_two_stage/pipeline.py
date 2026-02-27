"""
两阶段流水线：先检测目标，再对每个目标区域做精细分类。

流程:
    输入图片 → SSD 检测(定位目标) → 裁剪每个目标区域 → ResNet 分类(识别类别) → 合并输出

用法:
    python pipeline.py --image path/to/image.jpg
    python pipeline.py --image_dir path/to/folder/
    python pipeline.py --config configs/pipeline.yaml --image path/to/image.jpg
"""

import argparse
import json
import os

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from torchvision.transforms import functional as F

from models.detector import build_detector, AnchorGenerator
from models.model import build_model
from datasets.voc_dataset import VOC_CLASSES
from utils.det_utils import decode_boxes, nms


# ======================== 类别映射 ========================

# CIFAR-10 类别
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# VOC 类别 → CIFAR-10 类别映射（None 表示无对应）
VOC_TO_CIFAR = {
    "aeroplane": "airplane",
    "bicycle": None,
    "bird": "bird",
    "boat": "ship",
    "bottle": None,
    "bus": "truck",
    "car": "automobile",
    "cat": "cat",
    "chair": None,
    "cow": None,
    "diningtable": None,
    "dog": "dog",
    "horse": "horse",
    "motorbike": None,
    "person": None,
    "pottedplant": None,
    "sheep": None,
    "sofa": None,
    "train": None,
    "tvmonitor": None,
}

COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
]


# ======================== 配置 ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Detection → Classification Pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--det_weights", type=str, default=None)
    parser.add_argument("--cls_weights", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument("--nms_threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    return cfg


# ======================== 模型加载 ========================

def load_detector(weights_path, device):
    """加载 SSD 检测模型"""
    ckpt = torch.load(weights_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    num_classes = ckpt_cfg.get("num_classes", 21)
    input_size = ckpt_cfg.get("det_input_size", 300)

    model = build_detector(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    anchor_gen = AnchorGenerator(image_size=input_size)
    anchors = anchor_gen.generate(device=device)

    return model, anchors, input_size


def load_classifier(weights_path, device):
    """加载分类模型"""
    ckpt = torch.load(weights_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    model_name = ckpt_cfg.get("model_name", "resnet18")
    num_classes = ckpt_cfg.get("num_classes", 10)

    model = build_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


# ======================== 阶段一：检测 ========================

def det_preprocess(image, input_size):
    """检测模型预处理（ImageNet 归一化）"""
    img = F.resize(image, (input_size, input_size))
    img = F.to_tensor(img)
    img = T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(img)
    return img.unsqueeze(0)


@torch.no_grad()
def run_detection(model, image, anchors, device, input_size=300,
                  conf_threshold=0.3, nms_threshold=0.45):
    """
    阶段一：在整张图上运行 SSD 检测。
    Returns:
        detections: list of dict {
            'voc_class': str, 'det_conf': float,
            'bbox': [xmin, ymin, xmax, ymax] (像素坐标)
        }
    """
    orig_w, orig_h = image.size
    input_tensor = det_preprocess(image, input_size).to(device)

    cls_preds, reg_preds = model(input_tensor)
    cls_probs = torch.softmax(cls_preds[0], dim=1)
    boxes = decode_boxes(reg_preds[0], anchors).clamp(0, 1)

    detections = []
    num_classes = cls_probs.size(1)

    for cls_id in range(1, num_classes):  # 跳过背景
        scores = cls_probs[:, cls_id]
        mask = scores > conf_threshold
        if mask.sum() == 0:
            continue

        cls_scores = scores[mask]
        cls_boxes = boxes[mask]
        keep = nms(cls_boxes, cls_scores, nms_threshold)

        for idx in keep:
            box = cls_boxes[idx]
            detections.append({
                "voc_class": VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES)
                             else f"class_{cls_id}",
                "det_conf": cls_scores[idx].item(),
                "bbox": [
                    box[0].item() * orig_w,
                    box[1].item() * orig_h,
                    box[2].item() * orig_w,
                    box[3].item() * orig_h,
                ],
            })

    detections.sort(key=lambda x: x["det_conf"], reverse=True)
    return detections


# ======================== 阶段二：裁剪 + 分类 ========================

def cls_preprocess(crop, input_size=32):
    """分类模型预处理（CIFAR-10 归一化）"""
    img = F.resize(crop, (input_size, input_size))
    img = F.to_tensor(img)
    img = T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                      std=[0.2470, 0.2435, 0.2616])(img)
    return img


@torch.no_grad()
def crop_and_classify(classifier, image, detections, device,
                      cls_input_size=32, crop_padding=0.15,
                      min_crop_size=20, top_k=3):
    """
    阶段二：裁剪每个检测区域，用分类模型做精细分类。
    """
    orig_w, orig_h = image.size
    results = []

    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        box_w = xmax - xmin
        box_h = ymax - ymin

        # 跳过太小的框
        if box_w < min_crop_size or box_h < min_crop_size:
            results.append({
                **det,
                "cls_class": None,
                "cls_conf": 0.0,
                "cls_top_k": [],
                "skip_reason": "too_small",
            })
            continue

        # 外扩 padding 提供上下文
        pad_w = box_w * crop_padding
        pad_h = box_h * crop_padding
        crop_xmin = max(0, xmin - pad_w)
        crop_ymin = max(0, ymin - pad_h)
        crop_xmax = min(orig_w, xmax + pad_w)
        crop_ymax = min(orig_h, ymax + pad_h)

        # 裁剪
        crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        # 分类
        input_tensor = cls_preprocess(crop, cls_input_size).unsqueeze(0).to(device)
        output = classifier(input_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(min(top_k, probs.size(1)), dim=1)

        top_k_results = []
        for i in range(top_probs.size(1)):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            cls_name = CIFAR10_CLASSES[idx] if idx < len(CIFAR10_CLASSES) \
                       else f"class_{idx}"
            top_k_results.append({"class": cls_name, "conf": prob})

        # 检查 VOC→CIFAR 映射是否一致
        voc_cls = det["voc_class"]
        cifar_mapped = VOC_TO_CIFAR.get(voc_cls)
        cls_top1 = top_k_results[0]["class"] if top_k_results else None

        results.append({
            **det,
            "cls_class": cls_top1,
            "cls_conf": top_k_results[0]["conf"] if top_k_results else 0.0,
            "cls_top_k": top_k_results,
            "has_cifar_mapping": cifar_mapped is not None,
            "mapping_match": cifar_mapped == cls_top1 if cifar_mapped else None,
        })

    return results


# ======================== 可视化 ========================

def visualize(image, results, output_path=None):
    """在图片上绘制检测框 + 双标签（检测类别 | 分类类别）"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, r in enumerate(results):
        xmin, ymin, xmax, ymax = r["bbox"]
        color = COLORS[i % len(COLORS)]

        # 画框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # 标签：检测类别 | 分类类别
        det_label = f"{r['voc_class']} {r['det_conf']:.2f}"
        if r.get("cls_class"):
            cls_label = f"{r['cls_class']} {r['cls_conf']:.2f}"
            label = f"det:{det_label} | cls:{cls_label}"
        else:
            label = f"det:{det_label}"

        text_bbox = draw.textbbox((xmin, ymin - 15), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1,
             text_bbox[2] + 1, text_bbox[3] + 1],
            fill=color,
        )
        draw.text((xmin, ymin - 15), label, fill="white", font=font)

    if output_path:
        image.save(output_path)

    return image


# ======================== 主函数 ========================

def main():
    args = parse_args()
    cfg = load_config(args)

    # 设备
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # 加载模型
    det_weights = cfg.get("det_weights", "weights/detection/best_det.pt")
    cls_weights = cfg.get("cls_weights", "weights/voc_classification/best.pt")

    print(f"Loading detector from: {det_weights}")
    detector, anchors, det_input_size = load_detector(det_weights, device)

    print(f"Loading classifier from: {cls_weights}")
    classifier = load_classifier(cls_weights, device)

    print(f"Device: {device_str}")
    print()

    # 收集图片
    image_paths = []
    if cfg.get("image"):
        image_paths.append(cfg["image"])
    elif cfg.get("image_dir"):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(cfg["image_dir"])):
            if os.path.splitext(fname)[1].lower() in exts:
                image_paths.append(os.path.join(cfg["image_dir"], fname))

    if not image_paths:
        print("Error: Please provide --image or --image_dir")
        return

    # 输出目录
    output_dir = cfg.get("output_dir", "outputs/pipeline")
    os.makedirs(output_dir, exist_ok=True)

    conf_threshold = cfg.get("conf_threshold", 0.3)
    nms_threshold = cfg.get("nms_threshold", 0.45)
    cls_input_size = cfg.get("cls_input_size", 32)
    crop_padding = cfg.get("crop_padding", 0.15)
    min_crop_size = cfg.get("min_crop_size", 20)

    all_results = {}

    print(f"Processing {len(image_paths)} image(s)...")
    print("=" * 70)

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)

        # 阶段一：检测
        detections = run_detection(
            detector, image, anchors, device,
            input_size=det_input_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

        # 阶段二：裁剪 + 分类
        results = crop_and_classify(
            classifier, image, detections, device,
            cls_input_size=cls_input_size,
            crop_padding=crop_padding,
            min_crop_size=min_crop_size,
        )

        # 打印结果
        print(f"\n{img_name}  ({len(results)} objects detected)")
        print("-" * 70)
        for r in results:
            det_info = f"det: {r['voc_class']:>12s} {r['det_conf']:.3f}"
            if r.get("cls_class"):
                cls_info = f"cls: {r['cls_class']:>12s} {r['cls_conf']:.3f}"
                match = ""
                if r.get("has_cifar_mapping"):
                    match = " ✓" if r.get("mapping_match") else " ✗"
                print(f"  {det_info}  |  {cls_info}{match}")
            else:
                reason = r.get("skip_reason", "no_mapping")
                print(f"  {det_info}  |  (skipped: {reason})")

        # 可视化
        out_path = os.path.join(output_dir, img_name.rsplit(".", 1)[0] + "_pipeline.jpg")
        visualize(image.copy(), results, out_path)
        print(f"  Saved: {out_path}")

        # 收集 JSON 结果
        all_results[img_name] = [
            {k: v for k, v in r.items() if k != "cls_top_k"}
            for r in results
        ]

    # 保存 JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
