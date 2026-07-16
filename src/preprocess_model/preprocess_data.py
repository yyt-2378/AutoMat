import os
import parser
from .configs.option import args
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from preprocess_model.preprocess_dataset import DIVAESRDataset
from PIL import Image


class DIVAESRDataLoader(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.args = args

    def setup(self, stage: Optional[str] = None) -> None:
        for d in self.args.data_train:
            module_name = d

        self.train_dataset = DIVAESRDataset(self.args, name=module_name, split='training')
        self.val_dataset = DIVAESRDataset(self.args, name=module_name, split='validation')
        self.test_dataset = DIVAESRDataset(self.args, name=module_name, split='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
