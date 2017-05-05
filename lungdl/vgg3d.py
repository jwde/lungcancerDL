import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd.variable import Variable as Var

class Vgg3d(nn.Module):
    def __init__(self, weight_init=None):
        super(Vgg3d, self).__init__()
        self.init_layers(weight_init)

    def init_layers(self, weight_init):
        self.vggfeats = nn.Sequential(
            # Reducing only along XY
            nn.Conv3d(1, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),
            nn.Conv3d(64 , 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(128 , 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),
            )
        
        self.features = nn.Sequential(
                       #Now volumes are cubic... we can use actual 3d convs:w
            nn.Conv3d(128, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.scores = nn.Sequential(
            # per-instance logistic regression implemented as a 1x1 convolution
            # to elementwise sigmoid, to max pool
            nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=(7,7,7), stride=1, padding=0),
            #nn.Conv3d(128, 1, kernel_size=7, stride=1, padding=0),
        )
        self.probs = nn.Sigmoid()
        self.trainable = nn.Sequential(self.features, self.scores, self.probs)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #Xavier initialization
                var = None
                if weight_init:
                    var = weight_init
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
                    var = math.sqrt(2./n)
                m.weight.data.normal_(0, var)


    def forward(self, x):
        size = x.size()
        vggfeats = self.vggfeats(x)
        feats = self.features(vggfeats)
        scores = self.scores(feats)
        pred = self.probs(scores)
        pred = pred.view(size[0])
        return pred
def BGR_to_Grayscale(w, b, mean):
    new_w = torch.sum(w, 1)
    new_b = b
    w0 = w[:, 0, :, :] * mean[0]
    w1 = w[:, 1, :, :] * mean[1]
    w2 = w[:, 2, :, :] * mean[2]
    w_t = w0 + w1 + w2
    new_b -= torch.sum(torch.sum(w_t, 1), 2)
    return new_w, new_b

def to_3D(w):
    s = w.size()
    return w.view(s[0], s[1], 1, s[2], s[3])

def get_pretrained_2D_layers():
    from torchvision import models
    vgg = models.vgg16(pretrained=True)
    modules = vgg.features.modules()
    m = next(modules)
    mean = torch.FloatTensor([103.939, 116.779, 123.68])
    mean /= 255
    w0, b0 = m[0].weight.data, m[0].bias.data
    w1, b1 = to_3D(m[2].weight.data), m[2].bias.data
    w2, b2 = to_3D(m[5].weight.data), m[5].bias.data
    w3, b3 = to_3D(m[7].weight.data), m[7].bias.data
    w0, b0 = BGR_to_Grayscale(w0, b0, mean)
    w0 = to_3D(w0)

    cnn3d = Vgg3d()
    modules = next(cnn3d.vggfeats.modules())
    modules[0].weight.data = w0
    modules[0].bias.data = b0
    modules[2].weight.data = w1
    modules[2].bias.data = b1
    modules[5].weight.data = w2
    modules[5].bias.data = b2
    modules[7].weight.data = w3
    modules[7].bias.data = b3
    for p in cnn3d.vggfeats.parameters():
        p.requires_grad = False
    return cnn3d

def main():
    cnn3d = get_pretrained_2D_layers()
    for m in cnn3d.modules():
        if isinstance(m, nn.Conv3d):
            print (m.weight.size())
            print (m.bias.size())

    print("VGG")
    for m in Vgg3d().modules():
        if isinstance(m, nn.Conv3d):
            print (m.weight.size())
            print (m.bias.size())

if __name__ == '__main__':
    main()
