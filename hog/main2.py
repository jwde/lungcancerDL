#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import copy
import argparse
import os
from hw1.features import *
import sklearn.metrics as metrics

# Within package
import util

CONFIGS = {'alex3d' : {'net': lambda: 0}}
def main(r):
    # Disable interactive mode for matplotlib so docker wont segfault
    #plt.ioff()

    data_path = r.DATA_DIR
    labels_file = r.LABELS_FILE
    models_dir = r.MODELS_DIR
    save_name = r.SAVE_NAME
    load_name = r.LOAD_MODEL
    train_net = r.NET
    NUM_EPOCHS = r.NUM_EPOCHS
    training_size = r.TRAINING_SIZE
    use_validation = not r.NO_VAL

    config = CONFIGS[train_net]

    net = config['net']()
    crop = config.get('crop', None)
    lr = config.get('lr', 0.0001)
    reg = config.get('reg', 0.)
    batch_size = config.get('batch_size', 1)
    lr_scheduler = config.get('lr_scheduler', None)
    augment_data = config.get('augment_data', True)
    criterion = config.get('criterion', nn.BCELoss())
    get_probs = config.get('get_probs', None)
    xgboost = config.get('xgboost', False)
    max_depth = config.get('max_depth', 10)

    X = {'train' : [], 'val' : []}
    y = {'train' : [], 'val' : []}
    data = util.get_data(data_path, labels_file, batch_size, crop=crop, 
                         training_size=training_size,
                         augment_data=augment_data)
    for s in data:
        for i,d in enumerate(data[s]):
#            if i == 40:
#                break
            input, label = d
            input = input.numpy()[0,0,:,:,:, np.newaxis]
            label = int(label.numpy()[0][0])
            #print (input.shape, label)
            feature_fns = [hog_feature]
            feats = extract_features(input, feature_fns, verbose=True).flatten()
            X[s].append(feats)
            y[s].append(label)
        X[s] = np.array(X[s])
        y[s] = np.array(y[s])
    print (X['train'].shape, X['val'].shape)
    print (y['train'].shape, y['val'].shape)


    from sklearn.decomposition import PCA
    #from hw1.classifiers.linear_classifier import LinearSVM
    from sklearn.svm import SVC

    lr = 1e-5
    reg = 1e-3

    pca_dims = [16, 32, 64, 128, 256, 512, 1024]
    kernals = ['linear', 'rbf', 'sigmoid']
    results = {}
    best_val = -1
    best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
    USE_PCA = True
    if not USE_PCA:
        pca_dims = [0]
    for kernal in kernals:
        for d in pca_dims:
            if USE_PCA:
                pca = PCA(d)
                pca.fit(X['train'])
                #print (pca.explained_variance_ratio)
                x_train_transformed = pca.transform(X['train'])
                x_val_transformed   = pca.transform(X['val'])
            else:
                x_train_transformed = X['train']
                x_val_transformed   = X['val']
            print ("X_train shape", x_train_transformed.shape)
            svm = SVC(kernel=kernal, verbose = True)
            #loss_hist = svm.train(x_train_transformed, y['train'], learning_rate=lr, reg=reg,
            #              num_iters=10000, verbose=False)
            svm = svm.fit(x_train_transformed, y['train'])
            val_out = svm.predict(x_val_transformed)
            print(svm.score(x_val_transformed, y['val']))
            
            print(val_out, y['val'])
            val_acc = np.mean(svm.predict(x_val_transformed) == y['val'])
            val_auc = metrics.roc_auc_score(y['val'], val_out)
            val_ap  = metrics.average_precision_score(y['val'], val_out)
            val_precision = metrics.precision_score(y['val'], val_out)
            val_recall    = metrics.recall_score(y['val'], val_out)
            print(val_acc, val_auc, val_ap)
            results[(kernal,d)] = (np.mean(svm.predict(x_train_transformed) == y['train']), val_acc, val_precision, val_recall, val_auc, val_ap)
            if val_acc > best_val:
                best_val = val_acc
                best_svm = svm
                if USE_PCA:
                    best_pca = pca
                best_params = (d,) 
            print ("Val Accuracy:", val_acc)
            print ("Results", results[(kernal,d)])
    for k,v in results.items():
        kernel,d = k
        test_acc, acc, prec, rec, auc, ap =v 
        print(kernel, d,test_acc, acc, prec, rec, auc, ap)
    print (results)
################################################################################
#                              END OF YOUR CODE                                #
################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/a/data/lungdl/3Darrays_visual/', help="Path to data directory")
    parser.add_argument('--LABELS_FILE', default='/a/data/lungdl/cat_solutions.csv', help="Path to data directory")
    parser.add_argument('--NET', default='alex3d', help="One of: alex3d, 3d, simple")
    parser.add_argument('--MODELS_DIR', default='/a/data/lungdl/models/', help='Path to model directory')
    parser.add_argument('--SAVE_NAME', default='tmp.model', help='Name of save model')
    parser.add_argument('--LOAD_MODEL', default=None, help='Load pretrained model')
    parser.add_argument('--NUM_EPOCHS',  '-n', type=int, default=20, help='number of epochs to run')
    parser.add_argument('--TRAINING_SIZE',  '-s', type=int, default=644, help='number of')
    parser.add_argument('--NO_VAL', type=bool, nargs='?', const=True, default=False, help="Don't perform validation step")
    
    r = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: Cuda unavailable")
    main(r)
