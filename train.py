import torch
from dataset import ContrastiveLearningDataset
from simclr import SimCLR, SimCLR_ResNet
from utils import accuracy, save_checkpoint


from torch.cuda.amp import GradScaler, autocast

import os
import argparse
import wandb


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
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

    wandb.init(
        project='contrastive_learning_for_r3m'
    )
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = ContrastiveLearningDataset(root_folder='./dataset')
    training_dataset = dataset.get_dataset(name='cifar10', n_views=2)

    # define training and validation data loaders
    train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # get the model using our helper function
    model = SimCLR_ResNet()

    # move model to the right device
    model.to(device)

    simclr = SimCLR(model=model, args=args)

    # cnstruct an optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0,
                                                           last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    scaler = GradScaler(enabled=args.fp16_precision)


    n_iter = 0

    for epoch_counter in range(args.epochs):
        for images, _ in train_dataloader:
            images = torch.cat(images, dim=0)
            images = images.to(device)
            with autocast(enabled=args.fp16_precision):
                features = simclr.model(images)
                logits, labels = simclr.info_nce_loss(features)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if n_iter % args.log_every_n_steps == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                wandb.log({'iteration': n_iter,
                           'loss': loss,
                           'acc/top1': top1[0],
                           'acc/top5': top5[0],
                           'learning_rate': lr_scheduler.get_lr()[0]})
            n_iter += 1
        # warmup for the first 10 epochs
        if epoch_counter >= 10:
            lr_scheduler.step()
        wandb.log({'epoch': epoch_counter})
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch_counter)
        save_checkpoint({
            'epoch': args.epochs,
            'state_dict': simclr.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(os.getcwd(), 'model', checkpoint_name))
        wandb.save(os.path.join(os.getcwd(), 'model', checkpoint_name))
    

if __name__=='__main__':
    main()