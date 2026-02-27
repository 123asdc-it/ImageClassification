import torch.nn as nn


def build_loss(loss_type="cross_entropy", **kwargs):
    """
    构建损失函数。
    支持: cross_entropy, label_smoothing
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
