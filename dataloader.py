"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MnistBags(data_utils.Dataset):
    def __init__(self, train=True):
        self.train = train
        self.transform = self.ExtractPatches()
        self.dataset = datasets.MNIST(
            './datasets',
            train=self.train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,)),
                 self.transform]
                )
            )

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    class ExtractPatches(object):
        def __init__(self):
            pass

        def __call__(self, sample):
            sample = sample.unfold(1, 14, 14).unfold(2, 14, 14)
            sample = torch.permute(sample, (1, 2, 0, 3, 4))
            sample = torch.reshape(sample, (4, -1, 14, 14))
            return sample


if __name__ == "__main__":
    train_loader = data_utils.DataLoader(
        MnistBags(train=True),
        batch_size=1,
        shuffle=True
        )

    test_loader = data_utils.DataLoader(
        MnistBags(train=False),
        batch_size=1,
        shuffle=False
        )

    labels_train = np.zeros(10, dtype=int)
    for i, (bag, label) in enumerate(train_loader):
        labels_train[label] += 1
    print('Train bags: ', labels_train.sum())
    print(labels_train)

    labels_test = np.zeros(10, dtype=int)
    for i, (bag, label) in enumerate(test_loader):
        labels_test[label] += 1
        if i < 10:
            fig, axs = plt.subplots(2, 2, figsize=(5, 5))
            fig.subplots_adjust(hspace=0, wspace=0)
            for j in range(4):
                axs[j//2, j%2].axis('off')
                axs[j//2, j%2].imshow(torch.permute(bag[0][j], (1, 2, 0)))
            fig.savefig('./img/img_{}'.format(i))
    print('Test bags: ', labels_test.sum())
    print(labels_test)