import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd.variable import Variable as Var
from torchvision import models
import torch.nn as nn
def conv2d_to_bw(m, mean, std):
    # Sanity check...
    if not isinstance(m, nn.Conv2d): print ("Not Conv2d")
    if m.in_channels != 3:  print ("Not 3 input channels")
    
    # Modify weights and parameters
    m.in_channels = 1
    m.weight = m.weight.float()
    new_w = None
    new_b = None

    if m.bias is not None:
        new_w, new_b = weights_to_bw(m.weight.data, m.bias.data, mean, std)
        m.weight.data = new_w
        m.bias.data = new_b
    else:
        new_w, new_b = weights_to_bw(m.weight.data, None, mean, std)
        m.weight.data = new_w
        if new_b is not None:
            m.bias = nn.Parameter(new_b)

def weights_to_bw(w, bias = None, mean = None, std = [1., 1., 1.]):
    # Sum accross R,G,B channels
    w[:, 0, :, :].div_(std[0])
    w[:, 1, :, :].div_(std[1])
    w[:, 2, :, :].div_(std[2])
       
    new_w = torch.sum(w, 1)
    new_b = bias

    # Modify bias term as appropriate
    if mean != None:
        w0 = w[:, 0, :, :] * mean[0]
        w1 = w[:, 1, :, :] * mean[1]
        w2 = w[:, 2, :, :] * mean[2]
        w_t = w0 + w1 + w2
        extra_b = torch.sum(torch.sum(w_t, 1), 2)
        if bias is not None:
            new_b = bias - extra_b
        else:
            new_b = -extra_b
    if new_b is not None:
        N = new_b.size()[0]
        new_b = new_b.view(N)
    return new_w, new_b

def cnn_to_bw(cnn, mean = None, std = [1.,1.,1.]):
    first_conv2d = None

    # Find first Conv2d layer and break.
    for m in cnn.modules():
        if isinstance(m, nn.Conv2d):
            first_conv2d = m
            break
    conv2d_to_bw(first_conv2d, mean, std)

#### CODE for sanity checks...

# ALEXNET
# image | RGB class | BW class
# tiger | tiger     | zebra
# zebra | zebra     | zebra
# puzzle| puzzle    | puzzle
# german shep. | german shep. | weight scale?

import PIL.Image as Image
import numpy as np
from torch.autograd.variable import Variable as V
import torchvision.transforms as tt

def open_imagenet_image(path, size=(227,227), c_max=255.0):
    img = Image.open(path)
    img = np.asarray(img.resize(size))
    img = img.transpose(2,0,1)
    img = img / c_max
    img = torch.from_numpy(img).float()
    
    #Image net mean and standard deviations for pytorch models
    normalize = tt.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        std = [ 0.229, 0.224, 0.225 ]),
    img = normalize[0].__call__(img)
    return img

def main():
    net = models.vgg16(pretrained=True)
    #net = models.alexnet(pretrained=True).float().train(False)
    #    
    #cnn_to_bw(net)
    #for m in net.modules():
    #    print (m)
    imgpath = "img/tiger.jpg"
    img = open_imagenet_image(imgpath)
    img = img.view(1,3,227,227)
    pred = net.forward(V(img))
    print (torch.max(pred, 1))

    bw_img = open_imagenet_image(imgpath).mean(0)
    bw_img = bw_img.view(1,1,227,227)
    netbw = models.alexnet(pretrained=True).float().train(False)
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    cnn_to_bw(netbw, mean, std)
    pred = netbw.forward(V(bw_img))
    print(torch.max(pred,1))

if __name__ == '__main__':
    main()


