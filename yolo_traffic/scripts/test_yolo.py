"""
YOLOv8 测试/推理脚本

用法:
    python test_yolo.py --weights runs/yolo/train/weights/best.pt --source path/to/image.jpg
    python test_yolo.py --weights runs/yolo/train/weights/best.pt --source path/to/folder/
    python test_yolo.py --weights runs/yolo/train/weights/best.pt --source 0  # 摄像头
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="图片/文件夹/视频/摄像头")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/mps")
    parser.add_argument("--save", action="store_true", help="保存结果")
    parser.add_argument("--save_txt", action="store_true", help="保存标签文件")
    parser.add_argument("--save_conf", action="store_true", help="保存置信度")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument("--project", type=str, default="./outputs/predict", help="输出目录")
    parser.add_argument("--name", type=str, default="predict", help="实验名称")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设备处理
    device_str = args.device
    if device_str == "cuda":
        import torch
        if not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    print(f"Device: {device_str}")
    print(f"Source: {args.source}")
    print()

    # 推理
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device_str,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # 打印结果
    for i, r in enumerate(results):
        print(f"\nImage {i + 1}:")
        print(f"  Shape: {r.orig_shape}")
        print(f"  Detections: {len(r.boxes)}")

        if len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = r.names[cls_id]
                print(f"    {cls_name}: {conf:.3f}")

    print()
    print("=" * 70)
    print(f"Results saved to: {args.project}/{args.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
