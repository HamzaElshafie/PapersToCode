import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np

from labml_helpers.module import Module


def weights_init(module):
    """Initialise the weights of a given module.

    This function applies a normal distribution initialisation with mean 0.0 and
    standard deviation 0.02 to the weights of `Linear` layers in the module.

    Args:
        module (Module): The module whose weights need to be initialised.
    """
    class_name = module.__class__.__name
    if 'Linear' in class_name:
        with torch.no_grad():
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Generator(Module):
    """A generator model for generating images.

    This generator consists of a multi-layer perceptron (MLP) architecture with
    three hidden layers. Each hidden layer is followed by a LeakyReLU activation.
    The final layer produces an output image of shape (28, 28) using a `Tanh`
    activation function.

    The model initialises its weights using the `weights_init` function.

    Attributes:
        layers (nn.Sequential): A sequential container of linear layers and activation functions.
    """
    def __init__(self):
        """Initialise the generator network."""
        super().__init__()
        layer_sizes = [256, 512, 1024]
        layers = []
        prev_dim = 100

        for size in layer_sizes:
            layers = layers + [nn.Linear(prev_dim, size), nn.LeakyReLU(0.2)]
            prev_dim = size

        self.layers = nn.Sequential(*layers, nn.Linear(prev_dim, 28 * 28), nn.Tanh())

        # Initialise the weights of all the MLP layers
        self.apply(weights_init)

    def forward(self, x):
        """Forward pass of the generator.

        Args:
            x (Tensor): Input noise tensor of shape (batch_size, 100 (default)).

        Returns:
            Tensor: Generated image tensor of shape (batch_size, 1, 28, 28).
        """
        x = x.view(x.shape[0], 1, 28, 28)
        return self.layers(x)


class Discriminator(Module):
    """A discriminator model for distinguishing real and fake images.

    This discriminator consists of a multi-layer perceptron (MLP) architecture with
    three hidden layers. Each hidden layer is followed by a LeakyReLU activation.
    The final layer produces a single scalar logit, which can be interpreted as a
    raw score for determining whether the input image is real or fake.

    The model initializes its weights using the `weights_init` function.

    Attributes:
        layers (Sequential): A sequential container of linear layers and activation functions.
    """
    def __init__(self):
        super().__init__()
        layer_sizes = [1024, 512, 256]
        layers = []
        prev_dim = 28 * 28

        for size in layer_sizes:
            layers = layers + [nn.Linear(prev_dim, size), nn.LeakyReLU(0.2)]
            prev_dim = size

        self.layers = nn.Sequential(*layers, nn.Linear(prev_dim, 1))

        self.apply(weights_init)

    def forward(self, x):
        """Forward pass of the discriminator.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1), representing the
            raw score (logit) for each image. To obtain probabilities, apply
            a sigmoid activation function.
        """
        x = x.view(x.shape[0], -1)
        return self.layers(x)





