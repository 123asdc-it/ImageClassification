"""
两阶段 VOC 测试：先检测 → 再对每个框做 VOC 分类 → 可视化。

流程: 输入图片 → SSD 检测(定位) → 裁剪每个目标 → ResNet VOC 分类(21 类) → 画框+类别标签

用法:
    python test_two_stage_voc.py --image path/to/image.jpg --device mps
    # 默认权重：weights/detection/best_det.pt、weights/voc_classification/best.pt
    # 默认输出：outputs/two_stage_voc/
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision import transforms as T

from models.detector import build_detector, AnchorGenerator
from models.model import build_model
from datasets.voc_dataset import VOC_CLASSES
from utils.det_utils import decode_boxes, nms


COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
    "#80FF00", "#00FF80", "#800000", "#008000", "#000080",
    "#808000", "#800080", "#008080", "#FF4040", "#40FF40",
    "#4040FF",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage VOC: Detection → Classification → Visualize")
    # 默认使用项目根目录 weights/，转为绝对路径避免受 cwd 影响
    _root = os.path.abspath(os.path.dirname(__file__))
    default_det = os.path.join(_root, "weights", "detection", "best_det.pt")
    default_cls = os.path.join(_root, "weights", "voc_classification", "best.pt")
    parser.add_argument("--det_weights", type=str, default=default_det, help="检测模型权重，默认 weights/detection/best_det.pt")
    parser.add_argument("--cls_weights", type=str, default=default_cls, help="VOC 分类模型权重，默认 weights/voc_classification/best.pt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--det_input_size", type=int, default=300)
    parser.add_argument("--cls_input_size", type=int, default=224)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.4)
    parser.add_argument("--crop_padding", type=float, default=0.1)
    parser.add_argument("--min_crop_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(_root, "outputs", "two_stage_voc"),
                        help="可视化结果保存目录，默认 outputs/two_stage_voc")
    return parser.parse_args()


# ---------- 检测 ----------

def det_preprocess(image, input_size):
    img = F.resize(image, (input_size, input_size))
    img = F.to_tensor(img)
    img = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img)
    return img.unsqueeze(0)


def _iou(box1, box2):
    """计算两个框的 IoU，输入为 [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def _cross_class_nms(results, iou_threshold=0.5):
    """跨类别 NMS：去掉不同类别间高度重叠的低分框"""
    if len(results) <= 1:
        return results
    results = sorted(results, key=lambda x: x[1], reverse=True)
    keep = []
    for r in results:
        suppressed = False
        for k in keep:
            if _iou(r[2], k[2]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(r)
    return keep


@torch.no_grad()
def run_detection(model, image, anchors, device, input_size=300, conf_threshold=0.5, nms_threshold=0.4):
    """返回 list of (class_name, det_conf, [xmin, ymin, xmax, ymax]) 像素坐标"""
    orig_w, orig_h = image.size
    input_tensor = det_preprocess(image, input_size).to(device)

    cls_preds, reg_preds = model(input_tensor)
    cls_probs = torch.softmax(cls_preds[0], dim=1)
    boxes = decode_boxes(reg_preds[0], anchors).clamp(0, 1)

    results = []
    num_classes = cls_probs.size(1)
    for cls_id in range(1, num_classes):
        scores = cls_probs[:, cls_id]
        mask = scores > conf_threshold
        if mask.sum() == 0:
            continue
        cls_scores = scores[mask]
        cls_boxes = boxes[mask]
        keep = nms(cls_boxes, cls_scores, nms_threshold)
        for idx in keep:
            box = cls_boxes[idx]
            class_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f"class_{cls_id}"
            results.append((
                class_name,
                cls_scores[idx].item(),
                [
                    box[0].item() * orig_w,
                    box[1].item() * orig_h,
                    box[2].item() * orig_w,
                    box[3].item() * orig_h,
                ],
            ))

    # 跨类别 NMS，去掉不同类别间重叠的低分框
    results = _cross_class_nms(results, iou_threshold=0.5)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------- 分类（VOC 21 类）----------

def cls_preprocess(crop, input_size=224):
    """VOC 分类预处理：224、ImageNet 归一化"""
    img = F.resize(crop, (input_size, input_size))
    img = F.to_tensor(img)
    img = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img)
    return img.unsqueeze(0)


@torch.no_grad()
def classify_crops(classifier, image, det_results, device, cls_input_size=224,
                   crop_padding=0.1, min_crop_size=32):
    """
    对每个检测框裁剪后做 VOC 分类。
    det_results: list of (class_name, det_conf, bbox)
    返回: list of (det_class, det_conf, bbox, cls_class, cls_conf)
          cls_class 为 VOC 类别名（含 background），cls_conf 为分类置信度
    """
    orig_w, orig_h = image.size
    out = []

    for det_class, det_conf, bbox in det_results:
        xmin, ymin, xmax, ymax = bbox
        box_w = xmax - xmin
        box_h = ymax - ymin

        if box_w < min_crop_size or box_h < min_crop_size:
            out.append((det_class, det_conf, bbox, "__skip__", 0.0))
            continue

        pad_w = box_w * crop_padding
        pad_h = box_h * crop_padding
        cx1 = max(0, xmin - pad_w)
        cy1 = max(0, ymin - pad_h)
        cx2 = min(orig_w, xmax + pad_w)
        cy2 = min(orig_h, ymax + pad_h)
        crop = image.crop((cx1, cy1, cx2, cy2))

        tensor = cls_preprocess(crop, cls_input_size).to(device)
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)
        cls_conf, cls_idx = probs.max(dim=1)
        cls_idx = cls_idx.item()
        cls_conf = cls_conf.item()
        cls_class = VOC_CLASSES[cls_idx] if cls_idx < len(VOC_CLASSES) else f"class_{cls_idx}"

        # 融合策略：检测器高置信度时信任检测器
        if det_conf > 0.9 and cls_class != det_class:
            cls_class = det_class
            det_cls_idx = VOC_CLASSES.index(det_class) if det_class in VOC_CLASSES else -1
            det_cls_prob = probs[0][det_cls_idx].item() if det_cls_idx >= 0 else 0
            cls_conf = max(cls_conf, det_cls_prob)

        out.append((det_class, det_conf, bbox, cls_class, cls_conf))

    return out


# ---------- 可视化 ----------

def draw_results(image, results, output_path=None):
    """在图上绘制：框 + 标签「分类类别 + 分类置信度」(可选显示检测置信度)"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (IOError, OSError):
            font = ImageDraw.ImageFont.load_default()

    for i, (det_class, det_conf, bbox, cls_class, cls_conf) in enumerate(results):
        if cls_class == "__skip__":
            label = f"{det_class} (det:{det_conf:.2f}) [skip]"
            color = "#808080"
        else:
            label = f"{cls_class} {cls_conf:.2f}"
            if det_conf != cls_conf:
                label += f" (det:{det_conf:.2f})"
            cls_idx = VOC_CLASSES.index(cls_class) if cls_class in VOC_CLASSES else (i % len(VOC_CLASSES))
            color = COLORS[cls_idx % len(COLORS)]

        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1],
            fill=color,
        )
        draw.text((xmin, ymin), label, fill="white", font=font)

    if output_path:
        image.save(output_path)
        print(f"  Saved: {output_path}")
    return image


def main():
    args = parse_args()

    device_str = args.device or "cuda"
    if device_str == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # 加载检测模型
    print(f"Loading detector: {args.det_weights}")
    det_ckpt = torch.load(args.det_weights, map_location="cpu")
    det_cfg = det_ckpt.get("config", {})
    num_classes_det = det_cfg.get("num_classes", 21)
    detector = build_detector(num_classes=num_classes_det)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector = detector.to(device)
    detector.eval()

    det_input_size = det_cfg.get("det_input_size", args.det_input_size)
    anchor_gen = AnchorGenerator(image_size=det_input_size)
    anchors = anchor_gen.generate(device=device)
    print(f"  Detector: {num_classes_det} classes, input_size={det_input_size}")

    # 加载 VOC 分类模型（21 类）
    print(f"Loading classifier: {args.cls_weights}")
    cls_ckpt = torch.load(args.cls_weights, map_location="cpu", weights_only=False)
    cls_cfg = cls_ckpt.get("config", {})
    num_classes_cls = cls_cfg.get("num_classes", 21)
    model_name = cls_cfg.get("model_name", "resnet18")
    classifier = build_model(
        model_name=model_name,
        num_classes=num_classes_cls,
        pretrained=False,
    )
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()
    cls_input_size = cls_cfg.get("input_size", args.cls_input_size)
    print(f"  Classifier: {model_name}, {num_classes_cls} classes, input_size={cls_input_size}")

    # 图片列表
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in exts:
                image_paths.append(os.path.join(args.image_dir, fname))
    else:
        print("Error: Please provide --image or --image_dir")
        return

    if not image_paths:
        print("No images found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nDevice: {device_str}. Processing {len(image_paths)} image(s)...\n")
    print("-" * 70)

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        det_results = run_detection(
            detector, image, anchors, device,
            input_size=det_input_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
        )
        results = classify_crops(
            classifier, image, det_results, device,
            cls_input_size=cls_input_size,
            crop_padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )

        print(f"Image: {os.path.basename(img_path)}  ({len(results)} objects)")
        for det_class, det_conf, bbox, cls_class, cls_conf in results:
            if cls_class == "__skip__":
                print(f"  det:{det_class} {det_conf:.3f}  [skip small]")
            else:
                print(f"  det:{det_class} {det_conf:.3f}  →  cls:{cls_class} {cls_conf:.3f}  "
                      f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")

        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_two_stage.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        draw_results(image.copy(), results, out_path)
        print("-" * 70)

    print(f"Done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
