import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd.variable import Variable as Var
from torchvision import models

def main():
    #net = models.vgg16(pretrained=True)
    net = models.alexnet(pretrained=True)
    
    cnn_to_bw(net)
    for m in net.modules():
        print (m)

def conv2d_to_bw(m):
    # Sanity check...
    if not isinstance(m, nn.Conv2d): print ("Not Conv2d")
    if m.in_channels != 3:  print ("Not 3 input channels")
    
    # Modify weights and parameters
    m.in_channels = 1
    m.weight.data, m.bias.data = weights_to_bw(m.weight.data, m.bias.data)

def weights_to_bw(w, b, mean = None):
    # Sum accross R,G,B channels
    new_w = torch.sum(w, 1)
    new_b = b

    # Modify bias term as appropriate
    if mean != None:
        w0 = w[:, 0, :, :] * mean[0]
        w1 = w[:, 1, :, :] * mean[1]
        w2 = w[:, 2, :, :] * mean[2]
        w_t = w0 + w1 + w2
        new_b -= torch.sum(torch.sum(w_t, 1), 2)
    return new_w, new_b

def cnn_to_bw(cnn, mean = None):
    first_conv2d = None

    # Find first Conv2d layer and break.
    for m in cnn.modules():
        if isinstance(m, nn.Conv2d):
            first_conv2d = m
            break
    conv2d_to_bw(first_conv2d)

if __name__ == '__main__':
    main()


