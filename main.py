from DANN import DANN
from models import *
from dataloaders import *

import torch
import torch.nn as nn

from models import *
from dataloaders import *

e = Extractor_new().cuda()

source_loader = torch.utils.data.DataLoader(MNISTM(
    "../dataset/mnistm/",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=100, shuffle=True)


for index, src in enumerate(source_loader):
    data, label = src

    data = data.cuda()
    label.cuda()

    f = e(data)
