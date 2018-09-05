import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *


class DANN:
    def __init__(self, components, optimizers, dataloaderes, criterions, total_epoch, feature_dim, class_num, log_interval):

        self.extractor = components["extractor"]
        self.classifier = components["classifier"]
        self.discriminator = components["discriminator"]

        self.class_criterion = criterions["class"]
        self.domain_criterion = criterions["domain"]

        self.opt = optimizers["opt"]
        
        self.src_loader = dataloaderes["source_loader"]
        self.tar_loader = dataloaderes["target_loader"]

        self.total_epoch = total_epoch
        self.log_interval = log_interval
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.img_size = 28

    def train(self):
        print("[Training]")
        for epoch in range(self.total_epoch):
            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):
                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                """ For MNIST data, expand number of channel to 3 """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ train classifer """
                
                self.opt.zero_grad()

                src_z = self.extractor(src_data)
                tar_z = self.extractor(tar_data)

                pred_class = self.classifier(src_z)
                class_loss = self.class_criterion(pred_class, src_label)
                
                pred_d_src = self.discriminator(src_z)
                pred_d_tar = self.discriminator(tar_z)

                d_loss_src = self.domain_criterion(pred_d_src, torch.zeros(src_z.size(0)).type(torch.LongTensor).cuda())
                d_loss_tar = self.domain_criterion(pred_d_tar, torch.ones(tar_z.size(0)).type(torch.LongTensor).cuda())

                domain_loss = d_loss_src + d_loss_tar             
                
                loss = class_loss + domain_loss
                loss.backward()
                self.opt.step()

                if index % self.log_interval == 0:
                    print("[Epoch {:3d}] \t C_loss: {:.4f} \t D_loss:{:.4f}".format(epoch,
                                                                                class_loss, domain_loss))

    def test(self):
        print("[Testing]")

        self.extractor.eval()
        self.classifer.eval()
        self.discriminator.eval()

    def save_model(self, path="./saved_DANN/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.extractor, os.path.join(path, "DANN_E.pkl"))
        torch.save(self.classifier, os.path.join(path, "DANN_C.pkl"))
        torch.save(self.discriminator, os.path.join(path, "DANN_D.pkl"))

    def load_model(self, path="./saved_DANN/"):

        self.extractor.load_state_dict(torch.load(path, "DANN_E.pkl"))
        self.classifier.load_state_dict(torch.load(path, "DANN_C.pkl"))
        self.discriminator.load_state_dict(torch.load(path, "DANN_D.pkl"))

    def visualize(self, dim=2, plot_num=1000):
        print("t-SNE reduces to dimension {}".format(dim))

        self.extractor.cpu().eval()

        src_data = torch.FloatTensor()
        tar_data = torch.FloatTensor()

        ''' If use USPS dataset, change it to IntTensor() '''
        src_label = torch.LongTensor()
        tar_label = torch.IntTensor()

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

        src_z = self.extractor(src_data)
        tar_z = self.extractor(tar_data)

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
            visualize_2d(embedding, label, tag, self.class_num)

        elif dim == 3:
            visualize_3d(embedding, label, tag, self.class_num)