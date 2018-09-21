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
        self.classifier = components["classifier"]
        self.relater = components["relater"]
        self.discriminator = components["discriminator"]
        self.c_opt = optimizers["c_opt"]
        self.r_opt = optimizers["r_opt"]
        self.d_opt = optimizers["d_opt"]
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
        self.img_size = 28

    def train(self):

        for epoch in range(self.total_epoch):
            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):

                """ get data """
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
                self.c_opt.zero_grad()
                src_z = self.src_extractor(src_data)
                tar_z = self.tar_extractor(tar_data)

                pred_class = self.classifier(src_z)
                pred_loss = self.c_criterion(pred_class, src_label)

                _, predicted = torch.max(pred_class, 1)
                accuracy = 100.0 * \
                    (predicted == src_label).sum()/src_data.size(0)

                with torch.no_grad():
                    r = self.relater(src_z)
                    d_src_loss = self.discriminator(src_z)
                    d_tar_loss = self.discriminator(tar_z)

                #print("Classifier r.mean()= {}".format(r.mean()))
                w2_distance = (d_src_loss.mean() - d_tar_loss.mean())

                c_loss = pred_loss + r.mean()*w2_distance
                c_loss.backward()
                self.c_opt.step()

                """ train relater """
                self.r_opt.zero_grad()

                with torch.no_grad():
                    src_z = self.src_extractor(src_data)
                    tar_z = self.tar_extractor(tar_data)

                r_src = self.relater(src_z)
                r_tar = self.relater(tar_z)

                r_loss_src = self.r_criterion(r_src, torch.ones(
                    r_src.size(0), 1).type(torch.FloatTensor).cuda())

                r_loss_tar = self.r_criterion(r_tar, torch.zeros(
                    r_tar.size(0), 1).type(torch.FloatTensor).cuda())

                r_loss = r_loss_src + r_loss_tar
                r_loss.backward()
                self.r_opt.step()

                """ train discriminator """
                for _ in range(5):

                    with torch.no_grad():
                        r = self.relater(src_z)

                    gp = gradient_penalty(self.discriminator, src_z, tar_z)
                    d_src_loss = self.discriminator(src_z)
                    d_tar_loss = self.discriminator(tar_z)

                    #print("Discrimiator r.mean() = {}".format(r.mean()))
                    #d_src_loss *= r
                    w2_distance = (d_src_loss.mean() - d_tar_loss.mean())

                    d_loss = -r.mean()*w2_distance + 10*gp
                    d_loss.backward()
                    self.d_opt.step()

                total_loss = c_loss + r_loss + d_loss

                if index % self.log_interval == 0:
                    print("[Epoch {:3d}] Total_loss:{:.4f}   C_loss:{:.4f}   R_loss:{:.4f}   D_loss:{:.4f}".format
                          (epoch, total_loss, c_loss, r_loss, d_loss))

                    print("Classifier Accuracy: {:.2f}\n".format(accuracy))

                    # print("r_src {}".format(r_src))

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
            torch.load(os.path.join(path, "WADA_E_SRC.pkl")))

        self.tar_extractor.load_state_dict(
            torch.load(os.path.join(path, "WADA_E_TAR.pkl")))

        self.relater.load_state_dict(
            torch.load(os.path.join(path, "WADA_R.pkl")))

        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "WADA_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "WADA_D.pkl")))

    def visualize(self, dim, plot_num):
        print("t-SNE reduces to dimension {}".format(dim))

        self.src_extractor.cpu().eval()
        self.tar_extractor.cpu().eval()

        src_data = torch.FloatTensor()
        tar_data = torch.FloatTensor()

        src_label = torch.LongTensor()
        tar_label = torch.LongTensor()

        for index, src in enumerate(self.src_loader):
            data, label = src
            src_data = torch.cat((src_data, data))
            src_label = torch.cat((src_label, label))

        for index, tar in enumerate(self.tar_loader):
            data, label = tar
            tar_data = torch.cat((tar_data, data))
            tar_label = torch.cat((tar_label, label))

        ''' for MNIST dataset '''
        if src_data.shape[1] != 3:
            src_data = src_data.expand(
                src_data.shape[0], 3, self.img_size, self.img_size)

        src_data, src_label = src_data[0:plot_num], src_label[0:plot_num]
        tar_data, tar_label = tar_data[0:plot_num], tar_label[0:plot_num]

        src_z = self.src_extractor(src_data)
        tar_z = self.tar_extractor(tar_data)

        data = np.concatenate((src_z.detach().numpy(), tar_z.detach().numpy()))
        label = np.concatenate((src_label.numpy(), tar_label.numpy()))

        src_tag = torch.zeros(src_z.size(0))
        tar_tag = torch.ones(tar_z.size(0))
        tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

        ''' t-SNE process '''
        tsne = TSNE(n_components=dim)

        embedding = tsne.fit_transform(data)
        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min) / (embedding_max - embedding_min)

        if dim == 2:
            visualize_2d("./saved_WADA/", embedding,
                         label, tag, self.class_num)

        elif dim == 3:
            visualize_3d("./saved_WADA/", embedding,
                         label, tag, self.class_num)


if __name__ == "__main__":
    ''' paramters '''
    batch_size = 100
    total_epoch = 100
    feature_dim = 1000
    class_num = 10
    log_interval = 10
    test_batch_size = 100

    ''' dataloaders '''
    source_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=batch_size, shuffle=True)

    target_loader = torch.utils.data.DataLoader(MNISTM(
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

    test_tar_loader = torch.utils.data.DataLoader(MNISTM(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=False),  batch_size=test_batch_size, shuffle=True)

    ''' model components '''
    src_extractor = Extractor_new(encoded_dim=feature_dim).cuda()
    tar_extractor = Extractor_new(encoded_dim=feature_dim).cuda()
    relater = Relater(encoded_dim=feature_dim).cuda()
    classifier = Classifier(encoded_dim=feature_dim).cuda()
    discriminator = Discriminator_WGAN(encoded_dim=feature_dim).cuda()

    ''' optimizers '''
    """
    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": src_extractor.parameters()}], lr=1e-3)
    r_opt = torch.optim.Adam(relater.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam([{"params": discriminator.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-3)
    """
    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": src_extractor.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-3)
    r_opt = torch.optim.Adam(relater.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    ''' criterions '''
    c_criterion = nn.CrossEntropyLoss()
    r_criterion = nn.BCELoss()
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
    model.save_model()
    model.load_model()
    model.visualize(dim=2, plot_num=2000)
    model.test()
