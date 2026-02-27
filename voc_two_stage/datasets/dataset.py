from torchvision.datasets import CIFAR10
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms, MosaicTransform


class CIFAR10Dataset(CIFAR10):
    """
    扩展 CIFAR10，支持 Mosaic 数据增强。
    当 use_mosaic=True 时，以一定概率对样本做 Mosaic 拼接。
    """

    def __init__(self, root, train=True, download=True, transform=None,
                 use_mosaic=False, mosaic_prob=0.5, input_size=32):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.use_mosaic = use_mosaic and train
        self.mosaic_prob = mosaic_prob
        self.mosaic_transform = None
        if self.use_mosaic:
            self.mosaic_transform = MosaicTransform(self, input_size)

    def get_raw(self, index):
        """返回原始 PIL Image 和 label（不做 transform）"""
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        return img, target

    def __getitem__(self, index):
        import random
        if self.use_mosaic and random.random() < self.mosaic_prob:
            img, target = self.mosaic_transform(index)
        else:
            img, target = self.get_raw(index)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def build_dataset(cfg, is_train=True):
    """根据配置构建数据集"""
    data_root = cfg["data_root"]
    input_size = cfg.get("input_size", 32)
    use_mosaic = cfg.get("use_mosaic", False)
    mosaic_prob = cfg.get("mosaic_prob", 0.5)

    if is_train:
        transform = get_train_transforms(input_size)
    else:
        transform = get_val_transforms(input_size)

    dataset = CIFAR10Dataset(
        root=data_root,
        train=is_train,
        download=True,
        transform=transform,
        use_mosaic=use_mosaic if is_train else False,
        mosaic_prob=mosaic_prob,
        input_size=input_size,
    )
    return dataset
