"""
训练结果可视化脚本

用法:
    # 可视化最新的训练结果
    python visualize.py

    # 可视化指定的训练结果
    python visualize.py --results_dir runs/yolo/train

    # 指定输出目录
    python visualize.py --results_dir runs/yolo/train --output_dir visualizations
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import visualize_training_results


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO training results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="runs/yolo/train",
        help="训练结果目录 (包含 results.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="可视化图表输出目录 (默认保存到 results_dir)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nAvailable training results:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for train_dir in runs_dir.rglob("train"):
                if (train_dir / "results.csv").exists():
                    print(f"  - {train_dir}")
        return

    if not (results_dir / "results.csv").exists():
        print(f"Error: results.csv not found in {results_dir}")
        return

    print(f"Visualizing training results from: {results_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None

    visualize_training_results(results_dir)

    print("\n✓ Visualization complete!")
    print(f"\nTo view TensorBoard logs (if available):")
    print(f"  tensorboard --logdir {results_dir}")


if __name__ == "__main__":
    main()
