import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(24 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ConvNet
        """

        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 24 * 8 * 8)
        x = self.fc(x)
        return x
