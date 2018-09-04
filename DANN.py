import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class DANN:
    def __init__(components, optimizers, dataloaderes, total_epoch, feature_dim, class_num):

        self.extractor = components["extractor"]
        self.classifier = components["classifier"]
        self.discriminator = components["discriminator"]

        self.c_opt = optimizers["class_opt"]
        self.d_opt = optimizers["domain_opt"]

        self.src_loader = dataloaderes["source_loader"]
        self.tar_loader = dataloaderes["target_loader"]

        self.total_epoch = total_epoch
        self.feature_dim = feature_dim
        self.class_num = class_num

    def train():
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

                

    def test():

    def save_model(path="./saved_models/"):
        

    def load_model(path= "./saved_models/):

    def visualize(dim=2):
