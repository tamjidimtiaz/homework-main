from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    A multi-layer perceptron network to fit the training data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.kaiming_normal_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = activation()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        initializer(self.fc1.weight)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        initializer(self.fc2.weight)


    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # Flatten inputs to 2D (if more than that)
        x = x.view(x.shape[0], -1)

        # Get activations of each layer
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.nn.functional.log_softmax(x, dim=1)

        return self.x
