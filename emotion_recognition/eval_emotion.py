"""
FER2013 情绪分类评估脚本

用法:
    python eval_emotion.py --weights /path/to/best.pt --device mps
    python eval_emotion.py --weights /path/to/best.pt --image /path/to/face.jpg
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

from utils import AverageMeter

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build_model(cfg):
    model_name = cfg.get("model_name", "resnet18")
    num_classes = cfg["num_classes"]
    in_channels = cfg.get("in_channels", 1)

    model_fn = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}.get(model_name)
    model = model_fn(weights=None)

    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="FER2013 Emotion Evaluation")
    _root = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--weights", type=str,
                        default=os.path.join(_root, "weights", "best.pt"))
    parser.add_argument("--image", type=str, default=None, help="单张图片推理")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_root, "outputs"),
                        help="可视化结果保存目录，默认 outputs/")
    return parser.parse_args()


@torch.no_grad()
def evaluate_dataset(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(predicted, labels):
            cls_name = EMOTION_CLASSES[label.item()]
            class_total[cls_name] = class_total.get(cls_name, 0) + 1
            if pred.item() == label.item():
                class_correct[cls_name] = class_correct.get(cls_name, 0) + 1

    acc = 100.0 * correct / total
    return acc, class_correct, class_total, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def predict_image(model, image_path, cfg, device, output_dir=None):
    model.eval()
    input_size = cfg.get("input_size", 48)
    in_channels = cfg.get("in_channels", 1)

    tfm = []
    if in_channels == 1:
        tfm.append(transforms.Grayscale(num_output_channels=1))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm.extend([transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize])
    transform = transforms.Compose(tfm)

    img = Image.open(image_path).convert("RGB" if in_channels == 3 else "L")
    tensor = transform(img).unsqueeze(0).to(device)
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]

    print(f"\nImage: {image_path}")
    print("-" * 40)
    for i, (cls_name, prob) in enumerate(zip(EMOTION_CLASSES, probs)):
        bar = "#" * int(prob.item() * 30)
        print(f"  {cls_name:>10s}: {prob.item():.4f}  {bar}")
    top_idx = probs.argmax().item()
    print(f"\n  Prediction: {EMOTION_CLASSES[top_idx]} ({probs[top_idx].item():.2%})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        prob_values = [p.item() for p in probs]
        colors = ["#e74c3c" if i == top_idx else "#95a5a6" for i in range(len(EMOTION_CLASSES))]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]})

        # 左侧：原图
        axes[0].imshow(img, cmap="gray" if in_channels == 1 else None)
        axes[0].set_title(f"Prediction: {EMOTION_CLASSES[top_idx]} ({probs[top_idx].item():.1%})",
                          fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 右侧：概率条形图
        bars = axes[1].barh(EMOTION_CLASSES, prob_values, color=colors)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Probability")
        axes[1].invert_yaxis()
        for bar, val in zip(bars, prob_values):
            axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2%}", va="center", fontsize=9)

        plt.tight_layout()
        basename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{basename}_emotion.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


def save_eval_visualizations(acc, class_correct, class_total, all_preds, all_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(EMOTION_CLASSES)

    # 1. 混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix (Acc: {acc:.2f}%)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(EMOTION_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {cm_path}")

    # 2. 各类别准确率条形图
    accs = []
    for cls_name in EMOTION_CLASSES:
        total = class_total.get(cls_name, 0)
        correct = class_correct.get(cls_name, 0)
        accs.append(100.0 * correct / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    bars = ax.bar(EMOTION_CLASSES, accs, color=colors)
    ax.axhline(y=acc, color="red", linestyle="--", linewidth=1, label=f"Overall: {acc:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.legend()

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "per_class_accuracy.png")
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {acc_path}")


def main():
    args = parse_args()

    device_str = args.device or "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    print(f"Weights: {args.weights}")
    print(f"Epoch: {ckpt.get('epoch', '?')}, Best acc: {ckpt.get('best_acc', '?')}%")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.image:
        predict_image(model, args.image, cfg, device, output_dir=output_dir)
        return

    # 在测试集上评估
    data_root = cfg["data_root"]
    input_size = cfg.get("input_size", 48)
    in_channels = cfg.get("in_channels", 1)

    val_tfm = []
    if in_channels == 1:
        val_tfm.append(transforms.Grayscale(num_output_channels=1))
        val_tfm.extend([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5])])
    else:
        val_tfm.extend([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), transform=transforms.Compose(val_tfm))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    acc, class_correct, class_total, all_preds, all_labels = evaluate_dataset(model, val_loader, device)

    print(f"\nOverall accuracy: {acc:.2f}%")
    print(f"\nPer-class accuracy:")
    print("-" * 40)
    for cls_name in EMOTION_CLASSES:
        total = class_total.get(cls_name, 0)
        correct = class_correct.get(cls_name, 0)
        cls_acc = 100.0 * correct / total if total > 0 else 0
        print(f"  {cls_name:>10s}: {cls_acc:5.1f}%  ({correct}/{total})")

    # 保存可视化结果
    save_eval_visualizations(acc, class_correct, class_total, all_preds, all_labels, output_dir)


if __name__ == "__main__":
    main()
