import torch.nn as nn
import torch.nn.functional as F
import math

class Cnn3d(nn.Module):
    def __init__(self, weight_init=None):
        super(Cnn3d, self).__init__()
        self.init_layers(weight_init)

    def init_layers(self, weight_init):
        self.features = nn.Sequential(
            # Reducing only along XY
            nn.Conv3d(1, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),

            nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),

            #Now volumes are cubic... we can use actual 3d convs:w
            nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.predict = nn.Sequential(
            # per-instance logistic regression implemented as a 1x1 convolution
            # to elementwise sigmoid, to max pool
            nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(128, 128, kernel_size=1, stride=1, padding=0),
            #nn.MaxPool3d(kernel_size=(7,7,7), stride=1, padding=0)
            nn.Conv3d(128, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #Xavier initialization
                var = None
                if weight_init:
                    var = weight_init
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
                    var = math.sqrt(2./n)
                m.weight.data.normal_(0, var)


    def forward(self, x):
        size = x.size()
        x = self.features(x)
        pred = self.predict(x)
        pred = pred.view(size[0])
        return pred
