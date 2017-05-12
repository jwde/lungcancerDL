import random
import numpy as np
from hw1.data_utils import load_CIFAR10
#import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'


from hw1.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  # Load the raw CIFAR-10 data
  cifar10_dir = '/a/data/lungdl/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

from hw1.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

#PCA???
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train_feats)
pca_train = pca.transform(X_train_feats)
pca_val   = pca.transform(X_val_feats)
pca_test  = pca.transform(X_test_feats)

# Use the validation set to tune the learning rate and regularization strength

from hw1.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-5]
regularization_strengths = [1e-2]

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
def train(X_train_feats, y_train, X_val_feats, y_val, X_test_feats, y_test):
    best_val =-1 
    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = LinearSVM()
            loss_hist = svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg,
                          num_iters=10000, verbose=True)
            val_acc = np.mean(svm.predict(X_val_feats) == y_val)
            results[(lr, reg)] = (np.mean(svm.predict(X_train_feats) == y_train), val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_svm = svm
                best_params = lr, reg
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))

    print( 'best validation accuracy achieved during cross-validation: %f' % best_val)


# Evaluate your trained SVM on the test set
    y_test_pred = best_svm.predict(X_test_feats)
    test_accuracy = np.mean(y_test == y_test_pred)
    print (test_accuracy)
train(X_train_feats, y_train, X_val_feats, y_val, X_test_feats, y_test)
train(pca_train, y_train, pca_val, y_val, pca_test, y_test)
