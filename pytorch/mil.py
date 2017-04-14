import torch.nn as nn
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
