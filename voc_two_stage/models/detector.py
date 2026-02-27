"""
简易 SSD 风格目标检测器，使用 ResNet18 作为 backbone。

结构：
  ResNet18 backbone (去掉 avgpool + fc)
      ↓
  多尺度特征提取 (layer2, layer3, layer4)
      ↓
  检测头 (分类 + 回归) × 每个尺度
      ↓
  NMS 后处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import itertools
import math


class SSDDetector(nn.Module):
    """
    简易 SSD 检测器。
    - backbone: ResNet18 的 layer2/3/4 输出多尺度特征
    - 每个特征图位置预测 num_anchors 个框
    - 每个框预测 4 个坐标偏移 + num_classes 个类别分数
    """

    def __init__(self, num_classes=21, pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: ResNet18
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4,  64 channels
        self.layer2 = resnet.layer2  # stride 8,  128 channels
        self.layer3 = resnet.layer3  # stride 16, 256 channels
        self.layer4 = resnet.layer4  # stride 32, 512 channels

        # 额外的降采样层
        self.extra1 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 每个特征层的通道数和 anchor 数
        # 特征层: layer3(256), layer4(512), extra1(256), extra2(128)
        self.feat_channels = [256, 512, 256, 128]
        self.num_anchors = [6, 6, 6, 6]  # 每个位置的 anchor 数

        # 检测头：分类 + 回归
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for ch, na in zip(self.feat_channels, self.num_anchors):
            self.cls_heads.append(
                nn.Conv2d(ch, na * num_classes, 3, padding=1)
            )
            self.reg_heads.append(
                nn.Conv2d(ch, na * 4, 3, padding=1)
            )

        self._init_weights()

    def _init_weights(self):
        """初始化检测头权重"""
        for m in itertools.chain(self.cls_heads.modules(),
                                 self.reg_heads.modules(),
                                 self.extra1.modules(),
                                 self.extra2.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            cls_preds: (B, total_anchors, num_classes)
            reg_preds: (B, total_anchors, 4)
        """
        # Backbone 特征提取
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat3 = self.layer3(x)   # stride 16
        feat4 = self.layer4(feat3)  # stride 32
        feat5 = self.extra1(feat4)
        feat6 = self.extra2(feat5)

        features = [feat3, feat4, feat5, feat6]

        cls_preds = []
        reg_preds = []

        for feat, cls_head, reg_head in zip(features, self.cls_heads,
                                            self.reg_heads):
            B, _, H, W = feat.shape

            # 分类预测: (B, na*C, H, W) -> (B, H*W*na, C)
            cls = cls_head(feat)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(B, -1, self.num_classes)
            cls_preds.append(cls)

            # 回归预测: (B, na*4, H, W) -> (B, H*W*na, 4)
            reg = reg_head(feat)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            reg = reg.view(B, -1, 4)
            reg_preds.append(reg)

        # 拼接所有尺度
        cls_preds = torch.cat(cls_preds, dim=1)  # (B, total_anchors, C)
        reg_preds = torch.cat(reg_preds, dim=1)  # (B, total_anchors, 4)

        return cls_preds, reg_preds


class AnchorGenerator:
    """
    为 SSD 生成默认 anchor boxes。
    """

    def __init__(self, image_size=300):
        self.image_size = image_size
        # 重新设计 anchor 尺寸，覆盖 21px ~ 276px，匹配 VOC 目标分布
        self.feat_strides = [16, 32, 64, 128]
        self.anchor_sizes = [21, 45, 99, 153]
        self.anchor_ratios = [
            [2, 3, 0.5, 1/3],  # layer3: 6 anchors
            [2, 3, 0.5, 1/3],  # layer4: 6 anchors
            [2, 3, 0.5, 1/3],  # extra1: 6 anchors
            [2, 3, 0.5, 1/3],  # extra2: 6 anchors
        ]
        self.anchor_extra_sizes = [45, 99, 153, 207]

    def generate(self, device="cpu"):
        """
        生成所有 anchor boxes，格式为 (cx, cy, w, h)，归一化到 [0, 1]。
        Returns:
            anchors: (total_anchors, 4)
        """
        anchors = []
        s = self.image_size

        for stride, size, ratios, extra_size in zip(
            self.feat_strides, self.anchor_sizes,
            self.anchor_ratios, self.anchor_extra_sizes
        ):
            feat_size = math.ceil(s / stride)

            for i in range(feat_size):
                for j in range(feat_size):
                    cx = (j + 0.5) / feat_size
                    cy = (i + 0.5) / feat_size

                    # 基础 anchor
                    w = size / s
                    h = size / s
                    anchors.append([cx, cy, w, h])

                    # 额外的大 anchor
                    w2 = extra_size / s
                    h2 = extra_size / s
                    anchors.append([cx, cy, w2, h2])

                    # 不同宽高比的 anchor
                    for r in ratios:
                        w_r = size / s * math.sqrt(r)
                        h_r = size / s / math.sqrt(r)
                        anchors.append([cx, cy, w_r, h_r])

        return torch.tensor(anchors, dtype=torch.float32, device=device)


def build_detector(num_classes=21, pretrained=False):
    """构建 SSD 检测器"""
    return SSDDetector(num_classes=num_classes, pretrained=pretrained)
