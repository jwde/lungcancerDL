import torch.nn as nn
import torch.nn.functional as F
import math

class Cnn3d(nn.Module):
    def __init__(self):
        super(Cnn3d, self).__init__()
        self.init_layers()

    def init_layers(self):
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
           # nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
           # nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            #nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            #nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            #nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            #nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
           # nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)))
        self.predict = nn.Sequential(
            # per-instance logistic regression implemented as a 1x1 convolution
            # to elementwise sigmoid, to max pool
            nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool3d(kernel_size=(8,8,8), stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))


    def forward(self, x):
        size = x.size()
        x = self.features(x)
        pred = self.predict(x)
        pred = pred.view(size[0])
        return pred

