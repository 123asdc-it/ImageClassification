"""
目标检测推理脚本：对图片进行目标检测并可视化结果。

用法:
    python test_det.py --weights weights/detection/best_det.pt --image path/to/image.jpg
    python test_det.py --weights weights/detection/best_det.pt --image_dir path/to/folder/
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision import transforms as T

from models.detector import build_detector, AnchorGenerator
from datasets.voc_dataset import VOC_CLASSES
from utils.det_utils import decode_boxes, nms


# 每个类别一个颜色
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
    "#80FF00", "#00FF80", "#800000", "#008000", "#000080",
    "#808000", "#800080", "#008080", "#FF4040", "#40FF40",
    "#4040FF",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="置信度阈值")
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/detection/vis",
                        help="可视化结果保存目录，默认 outputs/detection/vis")
    return parser.parse_args()


def preprocess(image, input_size):
    """预处理图片"""
    img = F.resize(image, (input_size, input_size))
    img = F.to_tensor(img)
    img = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(img)
    return img.unsqueeze(0)


@torch.no_grad()
def detect(model, image, anchors, device, input_size=300,
           conf_threshold=0.3, nms_threshold=0.45):
    """
    对单张图片进行检测。
    Returns:
        results: list of (class_name, confidence, [xmin, ymin, xmax, ymax])
                 坐标为原图像素坐标
    """
    orig_w, orig_h = image.size
    input_tensor = preprocess(image, input_size).to(device)

    cls_preds, reg_preds = model(input_tensor)

    # 解码预测框
    cls_probs = torch.softmax(cls_preds[0], dim=1)  # (A, C)
    boxes = decode_boxes(reg_preds[0], anchors)       # (A, 4)

    # 裁剪到 [0, 1]
    boxes = boxes.clamp(0, 1)

    results = []
    num_classes = cls_probs.size(1)

    # 对每个类别（跳过背景类 0）做 NMS
    for cls_id in range(1, num_classes):
        scores = cls_probs[:, cls_id]
        mask = scores > conf_threshold

        if mask.sum() == 0:
            continue

        cls_scores = scores[mask]
        cls_boxes = boxes[mask]

        keep = nms(cls_boxes, cls_scores, nms_threshold)
        cls_scores = cls_scores[keep]
        cls_boxes = cls_boxes[keep]

        for score, box in zip(cls_scores, cls_boxes):
            # 转换为原图坐标
            xmin = box[0].item() * orig_w
            ymin = box[1].item() * orig_h
            xmax = box[2].item() * orig_w
            ymax = box[3].item() * orig_h

            class_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) \
                else f"class_{cls_id}"

            results.append((class_name, score.item(),
                            [xmin, ymin, xmax, ymax]))

    # 按置信度排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def draw_results(image, results, output_path=None):
    """在图片上绘制检测结果"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for cls_name, conf, bbox in results:
        xmin, ymin, xmax, ymax = bbox
        cls_idx = VOC_CLASSES.index(cls_name) if cls_name in VOC_CLASSES else 0
        color = COLORS[cls_idx % len(COLORS)]

        # 画框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # 画标签背景
        label = f"{cls_name} {conf:.2f}"
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1],
            fill=color,
        )
        draw.text((xmin, ymin), label, fill="white", font=font)

    if output_path:
        image.save(output_path)
        print(f"  Saved to: {output_path}")

    return image


def main():
    args = parse_args()

    # 设备
    device_str = args.device or "cuda"
    if device_str == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # 加载权重
    print(f"Loading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    num_classes = ckpt_cfg.get("num_classes", 21)

    # 构建模型
    model = build_detector(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Classes: {num_classes}, Device: {device_str}")

    # 生成 anchors
    anchor_gen = AnchorGenerator(image_size=args.input_size)
    anchors = anchor_gen.generate(device=device)

    # 收集图片
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

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nRunning detection on {len(image_paths)} image(s)...\n")
    print("-" * 60)

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        results = detect(
            model, image, anchors, device,
            input_size=args.input_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
        )

        print(f"Image: {os.path.basename(img_path)}  ({len(results)} detections)")
        for cls_name, conf, bbox in results:
            print(f"  {cls_name:>15s}  {conf:.3f}  "
                  f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

        # 可视化
        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        draw_results(image.copy(), results, out_path)

        print("-" * 60)


if __name__ == "__main__":
    main()
