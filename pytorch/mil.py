import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class MIL(nn.Module):
    def __init__(self, features):
        super(MIL, self).__init__()
        self.features = features
        self.mil = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.pred = nn.Sequential(
            nn.Sigmoid(),
            nn.MaxPool2d(7)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.mil(x)
        x_pred = self.pred(x)
        return x_pred, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MILCriterion(nn.Module):
    def __init__(self, w, sparsity, cuda=False):
        super(MILCriterion, self).__init__()
        self.crossent = nn.BCELoss()
        self.l1 = nn.L1Loss(size_average=False)
        self.l1_zeros = None
        self.w = w
        self.cuda = cuda
        self.sparsity = sparsity
    def forward(self, input, target):
        pred, mil = input
        loss = self.w * self.crossent(pred, target)
        if not self.l1_zeros:
            if self.cuda:
                self.l1_zeros = Variable(torch.FloatTensor(mil.size()).cuda(), requires_grad=False)
            else:
                self.l1_zeros = Variable(torch.FloatTensor(mil.size()), requires_grad=False)
        loss += self.sparsity * self.l1(mil, self.l1_zeros)
        return loss
