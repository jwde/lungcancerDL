import torch
from torch.autograd import Variable
import numpy as np
import xgboost as xgb
from cnn_to_grayscale import cnn_to_bw
IMGNET_MEAN= [ 0.485, 0.456, 0.406 ]
IMGNET_STD = [ 0.229, 0.224, 0.225 ]

class PretrainedFeaturesXGBoost(object):
    def __init__(self, extractor):
        cnn_to_bw(extractor, IMGNET_MEAN, IMGNET_STD)
        for param in extractor.parameters():
            param.requires_grad = False
        self.extractor = extractor
        if torch.cuda.is_available():
            self.extractor = self.extractor.cuda()
        self.tree = None

    def batch_features(self, xs):
        N, C, D, H, W = xs.size()

        xs.volatile = True
        feats = []

        # Forward pass each element through extractor to get slicewise features
        for x in xs:
            # We want (D, C, H, W) instead of (C, D, H, W)
            # Overloading the "batch" dimension so we do all slices at once
            slices = x.transpose(1,0)
            slice_feats = self.extractor(slices)

            # Now slice_feats are (N, C, H, W) or (60, 256, 6, 6)
            # So we transpose again
            slice_feats = slice_feats.transpose(0,1)
            feats.append(slice_feats)

        # collect features into one big tensor
        feats = torch.stack(feats)
        return feats.view(N, -1).data.cpu().numpy()

    def dataset_features(self, loader):
        features_np, labels_np = None, None
        for inputs, labels in loader:
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = self.batch_features(inputs)
            labels = labels.view(labels.size(0), -1).numpy()
            if features_np == None:
                features_np = outputs
                labels_np = labels
            else:
                features_np = np.concatenate((features_np, outputs), axis=0)
                labels_np = np.concatenate((labels_np, labels), axis=0)
        return features_np, labels_np

        
    def train(self, train_loader, val_loader, rounds, max_depth):
        param = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'eval_metric': ['error', 'auc', 'logloss']
        }
        
        print('Calculating train features...')
        train_features, train_labels = self.dataset_features(train_loader)
        print('Calculating val features...')
        val_features, val_labels = self.dataset_features(val_loader)

        print('Generating xgboost matrices...')
        train_matrix = xgb.DMatrix(train_features, label=train_labels)
        train_features, train_labels = None, None
        val_matrix = xgb.DMatrix(val_features, label=val_labels)
        val_features, val_labels = None, None

        eval_list = [(train_matrix, 'training'), (val_matrix, 'validation')]

        print('Training...')
        self.tree = xgb.train(param, train_matrix, rounds, eval_list)

    def __call__(self, xs):
        features = self.batch_features(xs)
        prediction = self.tree.predict(features)
        return prediction
