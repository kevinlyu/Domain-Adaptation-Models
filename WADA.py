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
        self.c_opt = optimizers["c_opt"]
        self.r_opt = optimizers["r_opt"]
        self.r_opt = optimizers["d_opt"]
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
                src_z = self.src_extractor(src_data)
                tar_z = self.tar_extractor(tar_data)

                pred_class = self.classifier(src_z)
                class_loss = self.c_criterion(pred_class, src_label)

                # wasserstein distance
                wasserstein_diatance = self.discriminator(
                    src_z).mean() - self.discriminator(tar_z).mean()

                # sliced wasserstein distance
                # sw =

                c_loss = class_loss + wasserstein_diatance

                """ classify accuracy """
                _, predicted = torch.max(pred_class, 1)
                accuracy = 100.0 * \
                    (predicted == src_label).sum() / src_data.size(0)

                self.c_opt.zero_grad()
                c_loss.backward()
                self.c_opt.step()

                """ train relater """
                with torch.no_grad():
                    # when train relator, do not bp to feature extractor
                    src_z = self.src_extractor(src_data)
                    tar_z = self.tar_extractor(tar_data)

                # let output in src --> 0 and target ---> 1 as partial transfer probability
                src_tag = torch.ones(src_z.size(0)).type(
                    torch.FloatTensor).cuda()
                tar_tag = torch.zeros(tar_z.size(0)).type(
                    torch.FloatTensor).cuda()

                # maximize the abilibity of relator to distinguish data domains
                r_pred_src = self.discriminator(src_z)
                r_pred_tar = self.discriminator(tar_z)

                r_loss_src = self.r_criterion(src_tag, r_pred_src)
                r_loss_tar = self.r_criterion(tar_tag, r_pred_tar)

                r_loss = r_loss_src + r_loss_tar
                r_opt.zero_grad()
                r_loss.backward()
                r_opt.step()

                """ train discriminator """
                for _ in range(5):

                    with torch.no_grad():
                        src_z = self.src_extractor(src_data)
                        tar_z = self.tar_extractor(tar_data)

                        r_src = self.relater(src_data)

                    gp = gradient_penalty(self.discriminator, src_z, tar_z)
                    d_src_loss = r_src*self.discriminator(src_z)
                    d_tar_loss = self.discriminator(tar_z)

                    wasserstein_distance = d_src_loss.mean()-d_tar_loss.mean()

                    domain_loss = -wasserstein_distance + 10*gp

                    d_opt.zero_grad()
                    domain_loss.backward()
                    d_opt.step()

                print(
                    "[Epoch {:3d}] Total_loss: {:.4f}\tC_loss: {:.4f}\tR_loss: {:.4f}\t")

    def test(self):
        print("[Testing]")

        self.src_extractor.cuda().eval()
        self.tar_extractor.cuda().eval()
        self.classifier.cuda().eval()

        src_correct = 0
        tar_correct = 0

        # testing source
        for index, src in enumerate(self.test_src_loader):
            src_data, src_label = src
            src_data, src_label = src_data.cuda(), src_label.cuda()

            ''' for MNIST '''
            if src_data.shape[1] != 3:
                src_data = src_data.expand(
                    src_data.shape[0], 3, self.img_size, self.img_size)

            src_z = self.src_extractor(src_data)
            src_pred = self.classifier(src_z)
            _, predicted = torch.max(src_pred, 1)
            src_correct += (predicted == src_label).sum().item()

        # testing target
        for index, (src, tar) in enumerate(zip(self.test_src_loader, self.test_tar_loader)):

            tar_data, tar_label = tar
            tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

            tar_z = self.tar_extractor(tar_data)
            tar_pred = self.classifier(tar_z)
            _, predicted = torch.max(tar_pred, 1)
            tar_correct += (predicted == tar_label).sum().item()

        print("source accuracy: {:.2f}%".format(
            100*src_correct/len(self.test_src_loader.dataset)))
        print("target accuracy: {:.2f}%".format(
            100*tar_correct/len(self.test_tar_loader.dataset)))

    def save_model(self, path="./saved_WADA/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.src_extractor.state_dict(),
                   os.path.join(path, "WADA_E_SRC.pkl"))

        torch.save(self.tar_extractor.state_dict(),
                   os.path.join(path, "WADA_E_TAR.pkl"))

        torch.save(self.relater.state_dict(), os.path.join(path, "WADA_R.pkl"))

        torch.save(self.classifier.state_dict(),
                   os.path.join(path, "WADA_C.pkl"))

        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, "WADA_D.pkl"))

    def load_model(self, path="./saved_WADA/"):

        self.src_extractor.load_state_dict(
            torch.load(os.paph.join(path, "WADA_E_SRC.pkl")))

        self.tar_extractor.load_state_dict(
            torch.load(os.paph.join(path, "WADA_E_TAR.pkl")))

        self.relater.load_state_dict(
            torch.load(os.path.join(path, "WADA_R.pkl")))

        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "WADA_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "WADA_D.pkl")))

    def visualize(self, dim):
        print("visualize to {}".format(dim))


if __name__ == "__main__":
    ''' paramters '''
    batch_size = 100
    total_epoch = 2
    feature_dim = 1000
    class_num = 10
    log_interval = 10
    test_batch_size = 3000

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
        ])), batch_size=test_batch_size, shuffle=True)

    test_tar_loader = torch.utils.data.DataLoader(USPS(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=False),  batch_size=test_batch_size, shuffle=True)

    ''' model components '''
    src_extractor = Extractor(encoded_dim=feature_dim)
    tar_extractor = Extractor(encoded_dim=feature_dim)
    relater = Relater(encoded_dim=feature_dim)
    classifier = Classifier(encoded_dim=feature_dim)
    discriminator = Discriminator_WGAN(encoded_dim=feature_dim)

    ''' optimizers '''
    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": src_extractor.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-3)
    r_opt = torch.optim.Adam(relater.parameters(), lr=1e-4)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    ''' criterions '''
    c_criterion = nn.BCELoss()
    r_criterion = nn.NLLLoss()
    # criterion of discriminator is defined as wasserstein by myself

    components = {"src_extractor": src_extractor, "tar_extractor": tar_extractor,
                  "relater": relater, "classifier": classifier, "discriminator": discriminator}
    dataloaders = {"src_loader": source_loader, "tar_loader": target_loader,
                   "test_src_loader": test_src_loader, "test_tar_loader": test_tar_loader}

    optimizers = {"c_opt": c_opt, "r_opt": r_opt, "d_opt": d_opt}

    criterions = {"c_criterion": c_criterion, "r_criterion": r_criterion}

    model = WADA(components, optimizers, dataloaders, criterions,
                 total_epoch, feature_dim, class_num, log_interval)
    model.train()
