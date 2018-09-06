import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *

class WADA:

    def __init__(self, components, optimizers, dataloaderes, criterions, total_epoch, feature_dim, class_num, log_interval):

        self.src_extractor = components["src_extractor"]
        self.tar_extractor = components["tar_extractor"]
        self.relater = components["relater"]
        self.discriminator = components["discriminator"]
        self.opt = optimizers["opt"]

        self.total_epoch = total_epoch
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.log_interval = log_interval
         
    