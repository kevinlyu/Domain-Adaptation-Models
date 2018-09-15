import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *
from models import *


class SAN:

    def __init__(self, components, optimizers, dataloaders, criterions, total_epoch, feature_dim, class_num, log_interval):

    def train(self):

    def test(self):

    def save_model():

    def load_model():

    def visualize():


if __name__ == "__main__":
    print("SAN model")

        batch_size = 150
    total_epoch = 300
    feature_dim = 1000
    class_num = 10
    log_interval = 10

    source_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=batch_size, shuffle=True)

    target_loader = torch.utils.data.DataLoader(USPS(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=True), batch_size=batch_size, shuffle=True)

    test_src_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=batch_size, shuffle=True)

    test_tar_loader = torch.utils.data.DataLoader(USPS(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=False),  batch_size=batch_size, shuffle=True)


    extractor = Extractor_new()
    classifier = Classifier()
    