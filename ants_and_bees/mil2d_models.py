from torchvision import models
import torch.nn as nn
import torch
import math

def xavier_init2d(modules):
     for m in modules:
         if isinstance(m, nn.Conv2d):
             n = m.kernel_size[0] * m.kernel_size[1]
             n *= m.in_channels
             var = math.sqrt(2./n)
             m.weight.data.normal_(0,var)

def alexnet256():
    extractor = models.alexnet(pretrained=True).float().features
    return simple_pretrained(extractor, 256, (6,6)).float()

def simple_pretrained(extractor, channels, features_shape, freeze=True):
    if freeze:
        for param in extractor.parameters():
            param.requires_grad = False
    predict = nn.Sequential(
        nn.Conv2d(channels, 1, features_shape, 1, 0),
        nn.Sigmoid())
    return Pretrained(extractor, predict)

class Pretrained(nn.Module):
    def __init__(self, features, predict):
       super(Pretrained, self).__init__()
       self.features = features # a pretrained features extractor
       self.predict = predict
       xavier_init2d(self.predict.modules())

    def forward(self, xs):
        # Forward pass feature extractor to extract
        N, C, H, W = xs.size()

        xs.volatile = True
        feats = self.features(xs)
        feats.volatile = False
        feats.requires_grad = True

        # Forward pass instance scorer and make prediction
        probs = self.predict(feats).view(N,-1)
        return probs

class PretrainedMIL(nn.Module):
    def __init__(self, features, mil_Scores):
        super(PretrainedMIL, self).__init__()
        self.features = features # a pretrained features extractor
        self.mil_scores = mil_scores
        xavier_init2d(self.mil_scores.modules())

    def forward(self, xs):
        N, C, H, W = xs.size()
        # Forward pass feature extractor to extract *Instances*
        xs.volatile = True
        feats = self.features(xs)
        feats.volatile = False
        feats.requires_grad = True

        # Forward pass instance scorer and make prediction
        scores = self.mil_scores(feats).view(N,-1)
        probs = nn.functional.sigmoid(scores.max(1)[0].view(N,-1))
        return probs, scores
