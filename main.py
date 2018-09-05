from DANN import DANN
from models import *
from dataloaders import *

import torch
import torch.nn as nn

batch_size = 100
total_epoch = 50
feature_dim = 500
class_num = 10
log_interval = 10


source_loader = torch.utils.data.DataLoader(datasets.MNIST(
    "../dataset/mnist/", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=True)


target_loader = torch.utils.data.DataLoader(MNISTM(
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])), batch_size=batch_size, shuffle=True)
'''
target_loader = torch.utils.data.DataLoader(USPS(
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])), batch_size=batch_size, shuffle=True)
'''

extractor = Extractor(encoded_dim=feature_dim).cuda()
classifier = Classifier(encoded_dim=feature_dim).cuda()
discriminator = Discriminator(encoded_dim=feature_dim).cuda()

class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()

'''
class_opt = torch.optim.Adam(classifier.parameters(), lr=1e-4)
domain_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
extractor_opt = torch.optim.Adam(extractor.parameters(), lr=1e-3)
'''

opt = torch.optim.Adam([{"params": classifier.parameters()},
                        {"params": extractor.parameters()},
                        {"params": discriminator.parameters()}], lr=1e-4)

components = {"extractor": extractor,
              "classifier": classifier, "discriminator": discriminator}
#optimizers = {"class_opt": class_opt, "domain_opt": domain_opt, "extractor_opt":extractor_opt}
optimizers = {"opt": opt}
dataloaders = {"source_loader": source_loader, "target_loader": target_loader}

criterions = {"class": class_criterion, "domain": domain_criterion}

model = DANN(components, optimizers, dataloaders,
             criterions, total_epoch, feature_dim, class_num, log_interval)
model.train()
model.save_model()
#model.load_model()
model.visualize(dim=2)
model.visualize(dim=3)
