import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    '''
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*0.5
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x):
        '''
        Extension of grad reverse layer
        '''
        return GradReverse.apply(x)


class Extractor(nn.Module):
    ''' Feature extractor '''

    def __init__(self, in_channels=16, lrelu_slope=0.2, encoded_dim=100):
        super(Extractor, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.encoded_dim = encoded_dim

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels*1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels*4),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_channels*4*7*7, self.encoded_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.extract(x)
        z = z.view(z.shape[0], self.in_channels*4*7*7)
        z = self.fc(z)

        return z


class Classifier(nn.Module):
    ''' Task Classifier '''

    def __init__(self, encoded_dim=100, class_num=10):
        super(Classifier, self).__init__()

        self.encoded_dim = encoded_dim
        self.class_num = class_num

        self.classify = nn.Sequential(
            nn.Linear(self.encoded_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.class_num),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.classify(x)


class Discriminator(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self, encoded_dim):
        super(Discriminator, self).__init__()
        self.encoded_dim = encoded_dim

        self.classify = nn.Sequential(
            nn.Linear(self.encoded_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = GradReverse.grad_reverse(x)
        return self.classify(x)
