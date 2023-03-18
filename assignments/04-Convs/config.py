from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 64
    num_epochs = 11

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=1e-4,
    )

    transforms = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
