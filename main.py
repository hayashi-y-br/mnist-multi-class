from __future__ import print_function

import argparse

import numpy as np
import torch
import torchvision.utils
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dataloader import MnistBags
from model import Attention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(MnistBags(train=True),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

print('Init Model')
if args.model=='attention':
    model = Attention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    loss_fn = torch.nn.CrossEntropyLoss()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        if args.cuda:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        X_batch, y_batch = Variable(X_batch), Variable(y_batch)

        optimizer.zero_grad()

        y_proba_list = []
        y_hat_list = []
        for j, (X, y) in enumerate(zip(X_batch, y_batch)):
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
            y_proba, y_hat, _ = model(X)
            y_proba_list.append(y_proba)
            y_hat_list.append(y_hat)
        y_proba = torch.cat(y_proba_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)

        loss = loss_fn(y_proba, y_batch)
        loss.backward()

        optimizer.step()

        train_loss += loss.detach().cpu().item()
        train_error += 1. - (y_hat == y_batch).detach().cpu().count_nonzero().item() / args.batch_size

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {:2d}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if args.cuda:
                X, y = X.cuda(), y.cuda()
            X, y = Variable(X), Variable(y)

            y_proba, y_hat, A = model(X)
            loss = loss_fn(y_proba, y)
            test_loss += loss.detach().cpu().item()
            test_error += 1. - (y_hat == y).detach().cpu().count_nonzero().item()

            if i < 100:
                X = X.detach().cpu()[0]
                A = A.detach().cpu()[0]
                save_result(X, A, title=f'$y = {y.detach().cpu().int()[0]}, \\hat{{y}} = {y_hat.detach().cpu().int()[0]}$', filename='fig_{}'.format(i))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))


def save_result(X, A, title=None, path='./img/', filename='img', mean=torch.tensor([0.3081]), std=torch.tensor([0.1307])):
    X = torchvision.utils.make_grid(X, nrow=2, padding=0)
    X = X * std + mean
    X = torch.permute(X, (1, 2, 0))
    A = A.view(2, 2)

    fig, ax = plt.subplots()
    if title is not None:
        fig.suptitle(title)
    ax.axis('off')
    ax.imshow(X)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(A, cmap='bwr', alpha=0.5, vmin=0., vmax=1., extent=[*xlim, *ylim])
    fig.savefig(path + filename)
    plt.close(fig)


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
