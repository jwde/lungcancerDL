from torchvision import models
import torch.nn as nn
from torch.autograd.variable import Variable as V
import torch
from cnn_to_grayscale import cnn_to_bw
import math
from xgboost_slicer import PretrainedFeaturesXGBoost

# Image net mean and std deviations for pytorch pretrained models
# (Used for RGB -> Grayscale conversion)
IMGNET_MEAN= [ 0.485, 0.456, 0.406 ]
IMGNET_STD = [ 0.229, 0.224, 0.225 ]

def xavier_init3d(modules):
    for m in modules:
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
            n *= m.in_channels
            var = math.sqrt(2./n)
            m.weight.data.normal_(0,var)

def simple_slicer(extractor, feature_dims, feature_size, freeze = True):
    cnn_to_bw(extractor, IMGNET_MEAN, IMGNET_STD)
    if freeze:
        for param in extractor.parameters():
            param.requires_grad = False
    predict = nn.Sequential(
        nn.Conv3d(feature_dims, 1, feature_size, 1, 0),
        nn.Sigmoid())
    return PretrainedSlicer(extractor, predict)

def simple_slicerMIL(extractor, feature_dims, freeze = True):
    cnn_to_bw(extractor, IMGNET_MEAN, IMGNET_STD)
    if freeze:
        for param in extractor.parameters():
            param.requires_grad = False
    mil_scores = nn.Sequential(
        nn.Conv3d(feature_dims, 1, 1, 1, 0))
    return PretrainedSlicerMIL(extractor, mil_scores)

def resnet_features(n, avg_pool=False):
    resnet = None
    if n == 18:
        resnet = models.resnet18(pretrained=True)
    elif n == 34:
        resnet = models.resnet34(pretrained=True)
    elif n == 50:
        resnet = models.resnet50(pretrained=True)
    elif n == 101:
        resnet = models.resnet101(pretrained=True)
    elif n == 152:
        resnet = models.resnet152(pretrained=True)
    else:
        print("WARNING: pretrained resnet{} does not exist".format(n))
        return
    features = None
    if avg_pool:
        features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AvgPool2d(8))
    else:
        features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4)
    return features

def ResNet(n):
    # resnet takes 227 x 227 and outputs 8x8x2048 in last layer
    extractor = resnet_features(n).float()
    return simple_slicer(extractor, 2048, (60, 8, 8)).float()
def ResNetMIL(n):
    extractor = resnet_features(n).float()
    return simple_slicerMIL(extractor, 2048).float()

def Alex(): 
    extractor = models.alexnet(pretrained=True).float().features
    return simple_slicer(extractor, 256, (60, 6, 6)).float()

def AlexMIL():
    extractor = models.alexnet(pretrained=True).float().features
    return simple_slicerMIL(extractor, 256).float()

# boosted models are not pytorch modules and must be trained by calling
# model.train(train_loader, val_loader, rounds, max_depth)
def ResNetBoosted(n):
    extractor = resnet_features(n, avg_pool=True).float()
    return PretrainedFeaturesXGBoost(extractor)

class PretrainedSlicerMIL(nn.Module):
    def __init__(self, features, mil_scores, weight_init=None):
        super(PretrainedSlicerMIL, self).__init__()
        self.features = features # a 2D convnet
        self.mil_scores = mil_scores
        xavier_init3d(self.mil_scores.modules())

    def forward(self, xs):
        N, C, D, H, W = xs.size()
        #exit()

        # Collect features for each element in batch
        xs.volatile = True
        feats = []

        # Forward pass each element through extractor to get slicewise features
        for x in xs:
            # We want (D, C, H, W) instead of (C, D, H, W)
            # Overloading the "batch" dimension so we do all slices at once
            slices = x.transpose(1,0)
            slice_feats = self.features(slices)

            # Now slice_feats are (N, C, H, W) or (60, 256, 6, 6)
            # So we transpose again
            slice_feats = slice_feats.transpose(0,1)
            feats.append(slice_feats)

        # collect features into one big tensor
        feats = torch.stack(feats)
        feats.volatile = False
        feats.requires_grad = True

        scores = self.mil_scores(feats).view(N,-1)
        probs = nn.functional.sigmoid(scores.max(1)[0].view(N,-1))
        return probs, scores

class PretrainedSlicer(nn.Module):
    def __init__(self, features, predict, weight_init=None):
        super(PretrainedSlicer, self).__init__()
        self.features = features # a 2D convnet
        self.predict = predict
        xavier_init3d(self.predict.modules())

    def forward(self, xs):
        N, C, D, H, W = xs.size()
        #exit()

        # Collect features for each element in batch
        xs.volatile = True
        feats = []

        # Forward pass each element through extractor to get slicewise features
        for x in xs:
            # We want (D, C, H, W) instead of (C, D, H, W)
            # Overloading the "batch" dimension so we do all slices at once
            slices = x.transpose(1,0)
            slice_feats = self.features(slices)

            # Now slice_feats are (N, C, H, W) or (60, 256, 6, 6)
            # So we transpose again
            slice_feats = slice_feats.transpose(0,1)
            feats.append(slice_feats)

        # collect features into one big tensor
        feats = torch.stack(feats)
        feats.volatile = False
        feats.requires_grad = True
        pred = self.predict(feats)
        return pred.view(N, -1) 


# Lives in the comments just in case...         
#class AlexSlicer(nn.Module):
#    def __init__(self, weight_init=None, freeze_alex=True):
#        super(AlexSlicer, self).__init__()
#
#        # Load pretrained alexnet and make it so it takes BW input
#        self.alexnet = models.alexnet(pretrained=True).float()
#        cnn_to_bw(self.alexnet, IMGNET_MEAN, IMGNET_STD)
#        
#        # freeze weights of pretrained net
#        if freeze_alex:
#            for param in self.alexnet.parameters():
#                param.requires_grad = False
#        self.init_layers(weight_init)
#
#    def init_layers(self, weight_init):
#        self.features = self.alexnet.features
#        self.predict = nn.Sequential(
#            nn.Conv3d(256, 1, kernel_size=(60,6,6), stride=1, padding=0),
#            nn.Sigmoid(), )
#        xavier_init3d(self.predict.modules())
#
#    def forward(self, xs):
#        xs = xs.float()
#        N, C, D, H, W = xs.size()
#
#        # Eventual prediction will have N probabilities
#        feats = []
#        for x in xs:
#            # We want (D, C, H, W) instead of (C, D, H, W)
#            # Overloading the "batch" dimension so we do all slices at once
#            slices = x.transpose(0,1)
#            slice_feats = self.features(slices)
#            # Now slice_feats are (N, C, H, W) or (60, 256, 6, 6)
#            # So we transpose again
#            slice_feats = slice_feats.transpose(0,1)
#            feats.append(slice_feats)
#        feats = torch.stack(feats)
#        pred = self.predict(feats)
#        return pred.view(N) 


