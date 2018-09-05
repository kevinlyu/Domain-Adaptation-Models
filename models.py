import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    '''
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x, constant):
        '''
        Extension of grad reverse layer
        '''
        return GradReverse.apply(x, constant)


class Extractor(nn.Module):
    ''' Feature extractor '''

    def __init__(self, in_channels=16, lrelu_slope=0.2, encoded_dim=100):
        super(Extractor, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.encoded_dim = encoded_dim

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels*1, kernel_size=5, padding=1),
            nn.BatchNorm2d(self.in_channels*1),
            nn.MaxPool2d(2),
            # nn.LeakyReLU(self.lrelu_slope),
            nn.ReLU(),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      4, kernel_size=5, padding=1),
            nn.BatchNorm2d(self.in_channels*4),
            # added
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            # nn.LeakyReLU(self.lrelu_slope)
        )

        '''
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels*4*5*5, self.encoded_dim),
            nn.ReLU()
        )
        '''

    def forward(self, x):
        z = self.extract(x)
        z = z.view(-1, 64*5*5)
        #z = self.fc(z)

        return z


class Classifier(nn.Module):
    ''' Task Classifier '''

    def __init__(self, encoded_dim=100, class_num=10):
        super(Classifier, self).__init__()

        self.encoded_dim = encoded_dim
        self.class_num = class_num

        self.classify = nn.Sequential(
            nn.Linear(64*5*5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # added
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.class_num),
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
            nn.Linear(64*5*5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(1)
        )

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        return self.classify(x)
