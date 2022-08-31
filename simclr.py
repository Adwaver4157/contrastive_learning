from typing import Any, Callable, List, Optional, Type, Union, overload

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from torch import Tensor

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SimCLR_ResNet(nn.Module):
    def __init__(self, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.feature_extractor = models.resnet50()
        self.latent_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()
        self.hidden_dim = hidden_dim
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
    
    def feature_extract(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        return x

class SimCLR():
    def __init__(self, model, args):
        self.model = model
        self.args = args
    
    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def train():
        raise NotImplementedError()

if __name__=='__main__':
    net = SimCLR_ResNet(64)
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
