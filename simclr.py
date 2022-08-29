from typing import Any, Callable, List, Optional, Type, Union, overload

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch import Tensor

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SimCLR_ResNet(nn.Module):
    def __init__(self, latent_dim, output_dim=256):
        super().__init__()
        self.feature_extractor = models.resnet50()
        self.latent_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()
        self.hidden_dim = latent_dim
        self.output_dim = output_dim
        self.prjection_head = nn.Sequential(
                                nn.Linear(self.latent_dim, self.hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_dim, self.output_dim)
                              ) 
    
    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature_extractor(x)
        x = self.prjection_head(x)

        return x

class SimCLR():
    def __init__(self, model):
        self.model = model
    
    def info_nce_loss(self, x):
        y = self.model(x)
    
    def train():
        raise NotImplementedError()

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])

if __name__=='__main__':
    net = SimCLR()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print(net)

    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
        # print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # print()

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
        # print(var_name, "\t", optimizer.state_dict()[var_name])
