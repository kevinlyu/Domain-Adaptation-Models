import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_office_loader(domain, partial, transform=None, target_transform=None):

    if domain == "amazon":
        loader = Amazon(partial=partial, transform=transform,
                        target_transform=target_transform)

    elif domain == "webcam":
        loader = Webcam(partial=partial, transform=transform,
                        target_transform=target_transform)

    elif domain == "dslr":
        loader = DSLR(partial=partial, transform=transform,
                      target_transform=target_transform)
    return loader


class Amazon(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            amazon = np.load(os.path.join(self.root, "amazon10.npz"))
        else:
            amazon = np.load(os.path.join(self.root, "amazon31.npz"))

        self.data, self.label = amazon["data"], amazon["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class Webcam(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            amazon = np.load(os.path.join(self.root, "webcam10.npz"))
        else:
            amazon = np.load(os.path.join(self.root, "webcam31.npz"))

        self.data, self.label = amazon["data"], amazon["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class DSLR(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            amazon = np.load(os.path.join(self.root, "dslr10.npz"))
        else:
            amazon = np.load(os.path.join(self.root, "dslr31.npz"))

        self.data, self.label = amazon["data"], amazon["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


if __name__ == "__main__":

    a = get_office_loader("amazon", partial=False)
    print(len(a))