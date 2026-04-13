import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='../utils', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='../utils', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

image, label = train_data[0]
print("\nImage Size in Dataset:")
print(image.size())

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch']

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        # 3 input channels (color options), 12 feature maps, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 12, 5) # (12, 28, 28)
        self.pool = nn.MaxPool2d(2, 2) # (12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5) # (24, 10, 10) -> (24, 5, 5)
        
        self.conv3 = nn.Conv2d(24, 48, 3) # (48, 3, 3)
        