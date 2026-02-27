import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """简单的 CNN 网络，用于 CIFAR-10 等小图分类"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name="resnet18", num_classes=10, pretrained=False):
    """
    根据名称构建模型。
    支持: resnet18, resnet34, resnet50, simple_cnn
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    # torchvision 预定义模型
    model_fn = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }.get(model_name)

    if model_fn is None:
        raise ValueError(f"Unknown model: {model_name}")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = model_fn(weights=weights)

    # 替换最后的全连接层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
