import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *
from models import *


class WADA:

    def __init__(self, components, optimizers, dataloaders, criterions, total_epoch, feature_dim, class_num, log_interval):

        self.src_extractor = components["src_extractor"]
        self.tar_extractor = components["tar_extractor"]
        self.relater = components["relater"]
        self.discriminator = components["discriminator"]
        self.opt = optimizers["opt"]

        self.src_loader = dataloaders["src_loader"]
        self.tar_loader = dataloaders["tar_loader"]
        self.test_src_loader = dataloaders["test_src_loader"]
        self.test_tar_loader = dataloaders["test_tar_loader"]

        self.c_criterion = criterions["c_criterion"]
        self.r_criterion = criterions["r_criterion"]

        self.total_epoch = total_epoch
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.log_interval = log_interval

    def train(self):

        for epoch in range(self.total_epoch):
            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):

                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                """ For MNIST """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ train classifier """
                

                """ train relater """

                """ train discriminator """

    def test(self):
        print("test WADA model")

    def save_model(self, path="./saved_WADA/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.extractor.state_dict(),
                   os.path.join(path, "WADA_E.pkl"))
        torch.save(self.classifier.state_dict(),
                   os.path.join(path, "WADA_C.pkl"))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, "WADA_D.pkl"))

    def load_model(self, path="./saved_WADA/"):

        self.extractor.load_state_dict(
            torch.load(os.path.join(path, "WADA_E.pkl")))
        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "WADA_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "WADA_D.pkl")))

    def visualize():


if __name__ == "__main__":
    ''' paramters '''
    batch_size = 100
    total_epoch = 20
    feature_dim = 1000
    class_num = 10
    log_interval = 10

    ''' dataloaders '''
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
        ])), batch_size=batch_size, shuffle=True)

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

    ''' model components '''
    src_extractor = Extractor(encoded_dim=feature_dim)
    tar_extractor = Extractor(encoded_dim=feature_dim)
    relater = Relater(encoded_num=feature_dim)
    classifier = Classifier(encoded_num=feature_im)
    discriminator = Discriminator_WGAN(encoded_dim=feature_dim)

    ''' optimizers '''
    c_opt = torch.optim.Adam([{"params": classifier.parameters(),
                              {"params": src_extractor.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-3)
    r_opt = torch.optim.Adam(relater.parameters(), lr=1e-4)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)


    ''' criterions '''
    c_criterion = nn.BCELoss()
    r_criterion = nn.NLLLoss()

    components = {"src_extractor": src_extractor, "tar_extractor": tar_extractor,
                  "relater": relater, "classifier": classifier, "discriminator": discriminator}
    dataloaders = {"src_loader": source_loader, "tar_loader": target_loader,
                   "test_src_loader": test_src_loader, "test_tar_loader": test_tar_loader}

    optimizers = {"c_opt": c_opt, "r_opt": r_opt, "d_opt": d_opt}

    criterions = {"c_criterion": c_criterion, "r_criterion": r_criterion}

    model = WADA(components, optimizers, dataloaders, criterions,
                 total_epoch, feature_dim, class_num, log_interval)
