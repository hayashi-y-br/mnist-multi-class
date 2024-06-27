import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

for i in range(10):
    img = []
    for j in range(10):
        img.append(plt.imread(f'./img_class/attention/img_{i}_{j}.png'))
    img = np.stack(img)
    img = torch.from_numpy(img)
    img = torch.permute(img, (0, 3, 1, 2))
    img = torchvision.utils.make_grid(img, nrow=5, padding=0)
    img = torch.permute(img, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    ax.imshow(img)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f'./result_class/attention_{i}')
    plt.close(fig)


for i in range(10):
    for k in range(10):
        img = []
        for j in range(10):
            img.append(plt.imread(f'./img_class/additive/img_{i}_{j}_{k}.png'))
        img = np.stack(img)
        img = torch.from_numpy(img)
        img = torch.permute(img, (0, 3, 1, 2))
        img = torchvision.utils.make_grid(img, nrow=5, padding=0)
        img = torch.permute(img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('off')
        ax.imshow(img)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(f'./result_class/additive_{i}_{k}')
        plt.close(fig)