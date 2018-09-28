import torch
import torch.nn as nn
from torch.autograd import grad


class Extractor_Office(nn.Module):

    def __init__(self, in_channels=128, lrelu_slope=0.02):
        super(Extractor_Office, self).__init__()

        self.lrelu_slope = lrelu_slope
        self.in_channels = in_channels

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels, 5),
            nn.BatchNorm2d(self.in_channels),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels, self.in_channels//2, 5),
            nn.BatchNorm2d(self.in_channels//2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels//2, self.in_channels//4, 3),
            nn.BatchNorm2d(self.in_channels//4),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels//4, self.in_channels//8, 3),
            nn.BatchNorm2d(self.in_channels//8),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
        )

    def forward(self, x):
        z = self.extract(x)
        z = z.view(-1, 16*4*4)
        return z


class Classifier_Office(nn.Module):

    def __init__(self, class_num):
        super(Classifier_Office, self).__init__()
        self.class_num = class_num

        self.classify = nn.Sequential(
            #nn.Linear(32*9*9, 100),
            nn.Linear(16*4*4, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            # added
            # nn.Dropout(),
            nn.Linear(50, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.class_num),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.classify(x)


class Discriminator_Office(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self):
        super(Discriminator_Office, self).__init__()

        self.classify = nn.Sequential(
            #nn.Linear(32*9*9, 64),
            nn.Linear(16*4*4, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classify(x)


class Relater_Office(nn.Module):

    ''' Relater network used in WADA model '''

    def __init__(self):
        super(Relater_Office, self).__init__()

        self.distinguish = nn.Sequential(
            #nn.Linear(32*9*9, 100),
            nn.Linear(16*4*4, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.distinguish(x)


def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
