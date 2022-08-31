import torch
from simclr import SimCLR, SimCLR_ResNet
from dataset import ContrastiveLearningDataset
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def main():
    args = parser.parse_args()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = ContrastiveLearningDataset(root_folder='./dataset')
    training_dataset = dataset.get_dataset(name='cifar10', n_views=2)

    # define training and validation data loaders
    dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model_path = os.path.join(os.getcwd(), 'model', 'checkpoint_0000.pt')
    model = SimCLR_ResNet().to(device)
    # simclr = SimCLR(model=model, args=args)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    x = np.array([])
    index = 0
    for images, _ in tqdm(dataloader):
        images = torch.cat(images, dim=0)
        images = images.to(device)
        features = model.feature_extract(images).to('cpu').detach().numpy().copy()
        if index == 0:
            x = features
        else:
            x = np.append(x, features, axis=0)
        index += 1
    
    import umap

    umap = umap.UMAP(n_components=2, random_state=0)
    X_reduced_umap = umap.fit_transform(x)
    plt.scatter(X_reduced_umap[:, 0], X_reduced_umap[:, 1], cmap='jet', alpha=0.5)
    plt.savefig('features_map.png')

if __name__=='__main__':
    main()