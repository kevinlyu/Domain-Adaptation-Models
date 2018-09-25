import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTM(Dataset):
    '''
    Definition of MNISTM dataset
    '''

    def __init__(self, root="/home/neo/dataset/mnistm/", train=True, partial=False, transform=None, target_transform=None):
        super(MNISTM, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.partial = partial

        if self.train:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "mnistm_pytorch_train"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_mnistm_pytorch_train"))
        else:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "mnistm_pytorch_test"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_mnistm_pytorch_test"))

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        # Return size of dataset
        return len(self.data)


class USPS(Dataset):
    '''
    Definition of USPS dataset
    '''

    def __init__(self, root="/home/neo/dataset/usps/", train=True, partial=False, transform=None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.partial = partial

        if self.train:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "usps_pytorch_train"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_usps_pytorch_train"))
        else:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "usps_pytorch_test"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_usps_pytorch_test"))

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data = np.stack([data]*3, axis=2)
        data = data.astype(float)
        data = Image.fromarray(np.uint8(data)*255, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.data)


########################################################################
# VisDA 2018
visda_syn_root = "../dataset/visda2018/train/"
VisdaSyn = torchvision.datasets.ImageFolder(visda_syn_root, transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

visda_syn_loader = torch.utils.data.DataLoader(
    VisdaSyn, batch_size=100, shuffle=True, num_workers=4)

visda_real_root = "../dataset/visda2018/validation/"

VisdaReal = torchvision.datasets.ImageFolder(visda_real_root, transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

visda_real_loader = torch.utils.data.DataLoader(
    VisdaReal, batch_size=100, shuffle=True, num_workers=4)


########################################################################

def get_office_loader(domain, class_num=31, train=True, partial=False):

    if class_num == 31:
        path = "../dataset/office31/"
    elif class_num == 10:
        path = "../dataset/office_caltech_10/"

    if domain == "amazon":
        Office_amazon = torchvision.datasets.ImageFolder(os.path.join(
            path, "amazon"), transform=transforms.Compose([
                transforms.Resize((120, 120)),
                transforms.CenterCrop((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

        loader = torch.utils.data.DataLoader(
            Office_amazon, batch_size=50, shuffle=True, num_workers=1)

    elif domain == "dslr":
        Office_dslr = torchvision.datasets.ImageFolder(os.path.join(
            path, "dslr"), transform=transforms.Compose([
                transforms.Resize((120, 120)),
                transforms.CenterCrop((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

        loader = torch.utils.data.DataLoader(
            Office_dslr, batch_size=50, shuffle=True, num_workers=1)

    elif domain == "webcam":
        Office_webcam = torchvision.datasets.ImageFolder(os.path.join(
            path, "webcam"), transform=transforms.Compose([
                transforms.Resize((120, 120)),
                transforms.CenterCrop((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

        loader = torch.utils.data.DataLoader(
            Office_webcam, batch_size=50, shuffle=True, num_workers=1)

    elif domain == "caltech":
        Caltech = torchvision.datasets.ImageFolder(os.path.join(
            path, "caltech"), transform=transforms.Compose([
                transforms.Resize((120, 120)),
                transforms.CenterCrop((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

        loader = torch.utils.data.DataLoader(
            Caltech, batch_size=50, shuffle=True, num_workers=1)

    return loader
