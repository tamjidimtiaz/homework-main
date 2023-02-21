from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomCrop,
)


class CONFIG:

    batch_size = 64
    num_epochs = 11
    initial_learning_rate = 0.01
    initial_weight_decay = 1e-04

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "T_max": 1280,
        "eta_min": 0.00001,
        "last_epoch": -1,
    }
    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=initial_weight_decay,
    )

    transforms = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
