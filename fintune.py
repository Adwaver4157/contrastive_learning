import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
from utils import accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=256,
                          num_workers=10, drop_last=False, shuffle=True)

test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=2*256,
                          num_workers=10, drop_last=False, shuffle=False)

model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
model_path = os.path.join(os.getcwd(), 'model', 'checkpoint_0200.pt')

state_dict = torch.load(model_path)['state_dict']

for k in list(state_dict.keys()):
  if k.startswith('feature_extractor.'):
    state_dict[k[len("feature_extractor."):]] = state_dict[k]
  del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.eval()

for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = 100
for epoch in range(epochs):
  top1_train_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(train_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    
    top1 = accuracy(logits, y_batch, topk=(1,))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  top1_train_accuracy /= (counter + 1)
  top1_accuracy = 0
  top5_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
  
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]
  
  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")