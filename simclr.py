from typing import Any, Callable, List, Optional, Type, Union, overload

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch import Tensor


class SimCLR_ResNet(nn.Module):
    def __init__(self, latent_dim, output_dim=256):
        super().__init__()
        self.feature_extractor = models.resnet50()
        self.latent_dim = self.feature_extractor.fc.in_features
        self.output_dim = output_dim
        self.prjection_head = nn.Sequential(
                                nn.Linear(self.latent_dim, self.hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_dim, self.output_dim)
                              ) 
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)

        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.prjection_head(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class SimCLR():
    def __init__(self):
        

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
