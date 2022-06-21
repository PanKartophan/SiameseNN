import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def plot_metric(loss, title='Loss'):
    plt.plot(loss['train'])
    plt.plot(loss['val'])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def calc_euclidean(x1, x2):
        return (x1 - x2).pow(2).sum(dim=1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class MNISTDataset(Dataset):
    def __init__(self, imgs, labels=None, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.imgs = imgs
            self.labels = labels
            self.indices = torch.tensor(range(len(labels)))
        else:
            self.imgs = imgs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        anchor_img = self.imgs[item].reshape(1, 28, 28)

        if self.is_train:
            anchor_label = self.labels[item]
            positive_indices = self.indices[self.indices != item][self.labels[self.indices != item] == anchor_label]

            positive_item = random.choice(positive_indices)
            positive_img = self.imgs[positive_item].reshape(1, 28, 28)

            negative_indices = self.indices[self.indices != item][self.labels[self.indices != item] != anchor_label]
            negative_item = random.choice(negative_indices)
            negative_img = self.imgs[negative_item].reshape(1, 28, 28)

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img
