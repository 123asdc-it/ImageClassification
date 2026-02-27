"""
从 COCO 2017 下载交通相关类别（person, bicycle, car, motorcycle, bus, truck），
转换为 YOLO 格式。

用法:
    python download_coco_traffic.py                          # 默认下载 5000 张
    python download_coco_traffic.py --max_images 2000        # 指定数量
    python download_coco_traffic.py --max_images 0           # 下载全部（约 6 万张）
    python download_coco_traffic.py --val_only               # 只下载验证集（约 2500 张，快速测试）
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

# COCO 交通相关类别 ID → 新 ID 映射
COCO_TRAFFIC_CLASSES = {
    1: 0,   # person     → 0
    2: 1,   # bicycle    → 1
    3: 2,   # car        → 2
    4: 3,   # motorcycle → 3
    6: 4,   # bus        → 4
    8: 5,   # truck      → 5
}

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

ANNO_URLS = {
    "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
ANNO_FILE = {
    "train": "annotations/instances_train2017.json",
    "val": "annotations/instances_val2017.json",
}


def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"  Downloading {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def download_annotations(data_dir):
    """下载 COCO 2017 标注（包含 train 和 val）"""
    zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
    anno_dir = os.path.join(data_dir, "annotations")

    if os.path.exists(os.path.join(anno_dir, "instances_train2017.json")) and \
       os.path.exists(os.path.join(anno_dir, "instances_val2017.json")):
        print("Annotations already exist, skipping download.")
        return

    download_file(ANNO_URLS["train"], zip_path)

    print("  Extracting annotations...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    print("  Done.")


def load_and_filter(anno_path, max_images=0):
    """加载 COCO 标注并筛选交通类别"""
    print(f"Loading {anno_path} ...")
    with open(anno_path, "r") as f:
        coco = json.load(f)

    target_cat_ids = set(COCO_TRAFFIC_CLASSES.keys())

    # 按图片分组标注
    img_annos = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids and ann["bbox"][2] > 1 and ann["bbox"][3] > 1:
            img_annos[ann["image_id"]].append(ann)

    # 只保留含有目标类别的图片
    valid_img_ids = set(img_annos.keys())
    images = [img for img in coco["images"] if img["id"] in valid_img_ids]
    print(f"  Found {len(images)} images with traffic objects.")

    if max_images > 0 and len(images) > max_images:
        random.seed(42)
        images = random.sample(images, max_images)
        print(f"  Sampled {max_images} images.")

    return images, img_annos


def download_images(images, out_img_dir, workers=8):
    """下载图片（多线程）"""
    os.makedirs(out_img_dir, exist_ok=True)

    to_download = []
    for img_info in images:
        dest = os.path.join(out_img_dir, img_info["file_name"])
        if not os.path.exists(dest):
            to_download.append((img_info["coco_url"], dest))

    if not to_download:
        print(f"  All {len(images)} images already downloaded.")
        return

    print(f"  Downloading {len(to_download)} images ({len(images) - len(to_download)} already exist)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = 0
    failed = 0

    def _download(url_dest):
        url, dest = url_dest
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, item): item for item in to_download}
        for future in as_completed(futures):
            if future.result():
                done += 1
            else:
                failed += 1
            total = done + failed
            if total % 200 == 0 or total == len(to_download):
                print(f"    [{total}/{len(to_download)}] downloaded={done}, failed={failed}")

    print(f"  Download complete: {done} success, {failed} failed.")


def convert_to_yolo(images, img_annos, out_label_dir):
    """将 COCO 标注转为 YOLO 格式"""
    os.makedirs(out_label_dir, exist_ok=True)

    for img_info in images:
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_id = img_info["id"]
        label_name = os.path.splitext(img_info["file_name"])[0] + ".txt"
        label_path = os.path.join(out_label_dir, label_name)

        lines = []
        for ann in img_annos[img_id]:
            coco_cat = ann["category_id"]
            if coco_cat not in COCO_TRAFFIC_CLASSES:
                continue

            new_cls = COCO_TRAFFIC_CLASSES[coco_cat]
            x, y, w, h = ann["bbox"]  # COCO: [x_min, y_min, width, height]

            # 转 YOLO: [x_center, y_center, width, height] 归一化
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))


def create_data_yaml(output_dir):
    """生成 YOLOv8 data.yaml"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck

nc: 6
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  data.yaml saved to {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Download COCO traffic subset in YOLO format")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
    parser.add_argument("--max_images", type=int, default=5000,
                        help="最大训练图片数，0=全部（约 6 万张）")
    parser.add_argument("--val_only", action="store_true",
                        help="只下载验证集（快速测试）")
    parser.add_argument("--workers", type=int, default=8,
                        help="下载线程数")
    args = parser.parse_args()

    output_dir = args.output_dir
    cache_dir = os.path.join(output_dir, "_cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("COCO Traffic Dataset Downloader")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Max train images: {'all' if args.max_images == 0 else args.max_images}")
    print("=" * 60)

    # 1. 下载标注
    print("\n[1/4] Downloading COCO 2017 annotations...")
    download_annotations(cache_dir)

    # 2. 筛选
    print("\n[2/4] Filtering traffic classes...")
    splits = {}
    if not args.val_only:
        train_anno = os.path.join(cache_dir, ANNO_FILE["train"])
        train_images, train_annos = load_and_filter(train_anno, max_images=args.max_images)
        splits["train"] = (train_images, train_annos)

    val_anno = os.path.join(cache_dir, ANNO_FILE["val"])
    val_images, val_annos = load_and_filter(val_anno, max_images=0)
    splits["val"] = (val_images, val_annos)

    # 3. 下载图片 + 转换标签
    print("\n[3/4] Downloading images & converting labels...")
    for split_name, (images, annos) in splits.items():
        print(f"\n  --- {split_name} ({len(images)} images) ---")
        img_dir = os.path.join(output_dir, "images", split_name)
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        download_images(images, img_dir, workers=args.workers)
        convert_to_yolo(images, annos, lbl_dir)

    # 4. 生成 data.yaml
    print("\n[4/4] Creating data.yaml...")
    yaml_path = create_data_yaml(output_dir)

    # 统计
    print("\n" + "=" * 60)
    print("Done!")
    for split_name, (images, _) in splits.items():
        print(f"  {split_name}: {len(images)} images")
    print(f"\n  Dataset: {output_dir}")
    print(f"  Config:  {yaml_path}")
    print(f"\n  Train command:")
    print(f"    yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
    print("=" * 60)


if __name__ == "__main__":
    main()
