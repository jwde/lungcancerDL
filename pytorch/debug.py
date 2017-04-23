import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class DebugModule(nn.Module):
    def __init__(self, module):
        super(DebugModule, self).__init__()
        self.module = module

    def forward(self, x):
        print("running module: {}".format(self.module))
        print("x shape: {}".format(x.size()))
        return self.module(x)
