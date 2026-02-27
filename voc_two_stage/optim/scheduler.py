import math
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, cfg):
    """
    构建学习率调度器，支持 warmup + cosine/linear/exp 衰减。

    Args:
        optimizer: torch.optim.Optimizer
        cfg: 配置字典，需包含 epochs, lr 等字段
    """
    warmup = cfg.get("warmup_epochs", 0)
    total = cfg["epochs"]
    min_lr = cfg.get("min_lr", 0)
    base_lr = cfg["lr"]
    sched_type = cfg.get("scheduler", "cosine")

    def lr_lambda(epoch):
        # warmup 阶段：线性从接近 0 升到 1
        if epoch < warmup:
            return max(1e-6 / base_lr, (epoch + 1) / warmup)

        # warmup 后的衰减
        progress = (epoch - warmup) / max(1, total - warmup)

        if sched_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif sched_type == "linear":
            factor = 1.0 - progress
        elif sched_type == "exp":
            factor = 0.95 ** (epoch - warmup)
        else:
            factor = 1.0

        return max(min_lr / base_lr, factor)

    return LambdaLR(optimizer, lr_lambda)
