from torch.utils.data import DataLoader


def build_dataloader(dataset, batch_size, num_workers, is_train=True):
    """构建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
