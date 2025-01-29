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
    class_name = module.__class.__name
    if 'Linear' in class_name:
        with torch.no_grad():
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Generator(Module):
    """

    """
    def __int__(self):
        super().__int__()
        layer_size = [256, 512, 1024]
        layers = []
        prev_dim = 100

        for size in layer_size:
            layers = layers + [nn.Linear(prev_dim, size), nn.LeakyReLU(0.2)]
            prev_dim = size

        self.layers = nn.Sequential(*layers, nn.Linear(prev_dim, 28 * 28), nn.Tanh())

        # Initialise the weights of all the MLP layers
        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x).view(x.shape[0], 1, 28, 28)


class Discriminator(nn.module):
    def __int__(self):
        pass

    def forward(self, x):
        pass

