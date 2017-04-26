import torch.nn as nn
import torch.nn.functional as F
import math


class WeightedBCELoss(nn.Module):
    def __init__(self, class_weight):
        self.class_weight = class_weight
        super(WeightedBCELoss, self).__init__()

    def forward(self, x, target):
        weight_tensor = self.class_weight * target.data + (1 - self.class_weight) * (1 - target.data)
        weight_tensor = weight_tensor.view(-1, 1)
        target = target.view(1, -1)
        return F.binary_cross_entropy(x, target, weight=weight_tensor)

