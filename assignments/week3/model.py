from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    build a multilayer perceptron model to fit the training data
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 2,
        activation: Callable = torch.nn.Sigmoid,
        initializer: Callable = torch.nn.init.ones_,
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

        for i in range(hidden_count):
            next_num_inputs = hidden_size[i]
            self.layers += [torch.nn.Linear(input_size, next_num_inputs)]
            initializer(self.layers[i].weight)
            input_size = next_num_inputs

        # Create final layer
        self.out = torch.nn.Linear(input_size, num_classes)

        # super().__init__()
        # self.activation = activation()
        # self.layers = torch.nn.ModuleList()
        # for i in range(len(hidden_size)):
        #   next_num_inputs = hidden_size[i]
        #   self.layers += [torch.nn.Linear(input_size, next_num_inputs)]
        #   input_size = next_num_inputs

        # # Create final layer
        # self.out = torch.nn.Linear(input_size, num_classes)

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
        for layer in self.layers:
            x = self.activation(layer(x))

        # Get outputs
        x = self.out(x)

        return torch.nn.functional.log_softmax(x, dim=1)
