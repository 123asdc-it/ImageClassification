"""
训练日志和可视化工具

功能:
- TensorBoard 日志记录
- 训练曲线可视化
- 学习率和损失分量分析
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingVisualizer:
    """训练结果可视化工具"""

    def __init__(self, results_dir):
        """
        Args:
            results_dir: 训练结果目录 (包含 results.csv)
        """
        self.results_dir = Path(results_dir)
        self.results_csv = self.results_dir / "results.csv"

        if not self.results_csv.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_csv}")

        self.df = pd.read_csv(self.results_csv)
        self.df.columns = self.df.columns.str.strip()  # 去除列名空格

    def plot_losses(self, save_path=None):
        """绘制损失曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation Losses', fontsize=16)

        # 训练损失
        ax = axes[0, 0]
        if 'train/box_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['train/box_loss'], label='Box Loss', marker='o')
        if 'train/cls_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['train/cls_loss'], label='Class Loss', marker='s')
        if 'train/dfl_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['train/dfl_loss'], label='DFL Loss', marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 验证损失
        ax = axes[0, 1]
        if 'val/box_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['val/box_loss'], label='Box Loss', marker='o')
        if 'val/cls_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['val/cls_loss'], label='Class Loss', marker='s')
        if 'val/dfl_loss' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['val/dfl_loss'], label='DFL Loss', marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 总损失对比
        ax = axes[1, 0]
        train_total = self.df.get('train/box_loss', 0) + self.df.get('train/cls_loss', 0) + self.df.get('train/dfl_loss', 0)
        val_total = self.df.get('val/box_loss', 0) + self.df.get('val/cls_loss', 0) + self.df.get('val/dfl_loss', 0)
        ax.plot(self.df['epoch'], train_total, label='Train Total', marker='o')
        ax.plot(self.df['epoch'], val_total, label='Val Total', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 损失分量占比 (最后一个epoch)
        ax = axes[1, 1]
        last_epoch = self.df.iloc[-1]
        losses = []
        labels = []
        if 'train/box_loss' in self.df.columns:
            losses.append(last_epoch['train/box_loss'])
            labels.append('Box Loss')
        if 'train/cls_loss' in self.df.columns:
            losses.append(last_epoch['train/cls_loss'])
            labels.append('Class Loss')
        if 'train/dfl_loss' in self.df.columns:
            losses.append(last_epoch['train/dfl_loss'])
            labels.append('DFL Loss')

        if losses:
            ax.pie(losses, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Loss Components (Epoch {int(last_epoch["epoch"])})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curves saved to: {save_path}")
        else:
            plt.savefig(self.results_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
            print(f"Loss curves saved to: {self.results_dir / 'loss_curves.png'}")

        plt.close()

    def plot_metrics(self, save_path=None):
        """绘制评估指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        # Precision & Recall
        ax = axes[0, 0]
        if 'metrics/precision(B)' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['metrics/precision(B)'], label='Precision', marker='o')
        if 'metrics/recall(B)' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['metrics/recall(B)'], label='Recall', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # mAP
        ax = axes[0, 1]
        if 'metrics/mAP50(B)' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['metrics/mAP50(B)'], label='mAP@0.5', marker='o')
        if 'metrics/mAP50-95(B)' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Learning Rate
        ax = axes[1, 0]
        lr_cols = [col for col in self.df.columns if col.startswith('lr/')]
        for col in lr_cols:
            ax.plot(self.df['epoch'], self.df[col], label=col.replace('lr/', ''), marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Training Time
        ax = axes[1, 1]
        if 'time' in self.df.columns:
            ax.plot(self.df['epoch'], self.df['time'], marker='o', color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Cumulative Training Time')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics curves saved to: {save_path}")
        else:
            plt.savefig(self.results_dir / "metrics_curves.png", dpi=300, bbox_inches='tight')
            print(f"Metrics curves saved to: {self.results_dir / 'metrics_curves.png'}")

        plt.close()

    def plot_learning_rate(self, save_path=None):
        """详细的学习率可视化"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Learning Rate Analysis', fontsize=16)

        # 学习率曲线
        ax = axes[0]
        lr_cols = [col for col in self.df.columns if col.startswith('lr/')]
        for col in lr_cols:
            ax.plot(self.df['epoch'], self.df[col], label=col.replace('lr/', ''), marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 学习率变化率
        ax = axes[1]
        for col in lr_cols:
            lr_change = self.df[col].diff()
            ax.plot(self.df['epoch'][1:], lr_change[1:], label=col.replace('lr/', ''), marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR Change')
        ax.set_title('Learning Rate Change Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate analysis saved to: {save_path}")
        else:
            plt.savefig(self.results_dir / "lr_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Learning rate analysis saved to: {self.results_dir / 'lr_analysis.png'}")

        plt.close()

    def generate_summary(self):
        """生成训练摘要"""
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)

        last_epoch = self.df.iloc[-1]
        best_map50_idx = self.df['metrics/mAP50(B)'].idxmax() if 'metrics/mAP50(B)' in self.df.columns else 0
        best_epoch = self.df.iloc[best_map50_idx]

        print(f"\nTotal Epochs: {int(last_epoch['epoch'])}")
        print(f"Total Training Time: {last_epoch['time']:.2f} seconds ({last_epoch['time']/60:.2f} minutes)")

        print(f"\nFinal Metrics (Epoch {int(last_epoch['epoch'])}):")
        if 'metrics/precision(B)' in self.df.columns:
            print(f"  Precision: {last_epoch['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in self.df.columns:
            print(f"  Recall: {last_epoch['metrics/recall(B)']:.4f}")
        if 'metrics/mAP50(B)' in self.df.columns:
            print(f"  mAP@0.5: {last_epoch['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in self.df.columns:
            print(f"  mAP@0.5:0.95: {last_epoch['metrics/mAP50-95(B)']:.4f}")

        print(f"\nBest mAP@0.5 (Epoch {int(best_epoch['epoch'])}):")
        if 'metrics/mAP50(B)' in self.df.columns:
            print(f"  mAP@0.5: {best_epoch['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in self.df.columns:
            print(f"  mAP@0.5:0.95: {best_epoch['metrics/mAP50-95(B)']:.4f}")

        print(f"\nFinal Losses:")
        if 'train/box_loss' in self.df.columns:
            print(f"  Train Box Loss: {last_epoch['train/box_loss']:.4f}")
        if 'train/cls_loss' in self.df.columns:
            print(f"  Train Class Loss: {last_epoch['train/cls_loss']:.4f}")
        if 'train/dfl_loss' in self.df.columns:
            print(f"  Train DFL Loss: {last_epoch['train/dfl_loss']:.4f}")

        print("=" * 70 + "\n")

    def visualize_all(self, output_dir=None):
        """生成所有可视化图表"""
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")
        self.plot_losses(output_dir / "loss_curves.png")
        self.plot_metrics(output_dir / "metrics_curves.png")
        self.plot_learning_rate(output_dir / "lr_analysis.png")
        self.generate_summary()
        print(f"\nAll visualizations saved to: {output_dir}")


def visualize_training_results(results_dir):
    """
    便捷函数：可视化训练结果

    Args:
        results_dir: 训练结果目录路径
    """
    visualizer = TrainingVisualizer(results_dir)
    visualizer.visualize_all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python logger.py <results_dir>")
        print("Example: python logger.py runs/yolo/train")
        sys.exit(1)

    results_dir = sys.argv[1]
    visualize_training_results(results_dir)
