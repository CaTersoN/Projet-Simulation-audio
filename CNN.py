import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super.__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,1,3, padding = (1,1))
        self.conv2 = nn.Conv2d(1,1,3, padding = (1,1))

    def forward(self, x):
        x = F.relu(self.conv1)
        x = F.relu(self.conv2)
        return x

Model = NeuralNetwork()
print(Model)


