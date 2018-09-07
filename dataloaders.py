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

    def __init__(self, root="/home/neo/dataset/mnistm/", train=True, transform=None, target_transform=None):
        super(MNISTM, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.data, self.label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_train"))
        else:
            self.data, self.label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_test"))

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

    def __init__(self, root="/home/neo/dataset/usps/", train=True, transform=None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.data, self.label = torch.load(
                os.path.join(self.root, "usps_pytorch_train"))
        else:
            self.data, self.label = torch.load(
                os.path.join(self.root, "usps_pytorch_test"))

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]

        """
        print("loaded image")
        plt.imshow(data)
        plt.show()
        """
        #data = np.reshape(data, (16, 16, 1))
        
        data = np.stack([data]*3, axis=2)
        data = data.astype(float)
        #print(np.shape(data))
        #plt.imshow(data)
        #plt.show()        
        data = Image.fromarray(np.uint8(data)*255, mode="RGB")
        #data = Image.fromarray(data, mode="RGB")
        #data.show()
        #exit()
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.data)
