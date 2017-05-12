import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import argparse
import os
import sys
import sklearn.metrics as metrics

# Within package
import models
import util
from config import CONFIGS


def evaluate_model(test_loader, model, criterion, get_probs=None):
    running_loss = 0.
    running_corrects = 0
    all_probs = []
    all_labels = []
    for i, (inputs, labels) in enumerate(test_loader):
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            
        # forward
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # statistics
        if get_probs:
            outputs = get_probs(outputs)
        #average_precision = metrics.average_precision_score(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
        
        running_loss += loss.data[0]
        running_corrects += torch.sum((outputs.data > .5) == (labels.data > .5))

        all_labels += [labels.data.cpu()]
        all_probs += [outputs.data.cpu()]

    # Note: BCE loss already divides by batchsize
    epoch_loss = running_loss / (len(test_loader))
    epoch_acc = float(running_corrects) / len(test_loader) 

    all_probs = torch.cat(all_probs, 0).view(-1).numpy()
    all_labels = torch.cat(all_labels, 0).view(-1).numpy()

    print('Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc)) 

    return get_statistics(all_labels, all_probs)

def get_statistics(all_labels, all_probs):
    avg_precision = metrics.average_precision_score(all_labels, all_probs, average="macro")
    weighted_ap = metrics.average_precision_score(all_labels, all_probs, average="weighted")
    pr_curve = metrics.precision_recall_curve(all_labels, all_probs)
    roc_curve = metrics.roc_curve(all_labels, all_probs)
    
    print('AP {:.4f} mAP: {:.4f}'.format(avg_precision, weighted_ap)) 
    return avg_precision, weighted_ap, pr_curve, roc_curve

def get_lung_data(lungs_dir, labels_file):
    dset_loaders = util.get_data(lungs_dir, labels_file, 1, training_size = 0, augment_data = False)
    test_loader = dset_loaders['val']
    return test_loader

def plot_pr(pr_curve, model_name, figs_dir):
    plt.clf()
    precision, recall, _ = pr_curve
    plt.plot(recall, precision, lw=2, color='black', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0., 1.05])
    plt.xlim([0., 1.05])
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(figs_dir, '{}_pr.png'.format(model_name)))
    return
def plot_roc(roc_curve):
    return

def evaluate_dumb_model(p, lungs_dir, labels_file):
    print("-"*10)
    print("Probability all {}".format(p))
    test_loader = get_lung_data(lungs_dir, labels_file)
    all_labels = []
    for data, labels in test_loader:
        all_labels += [labels]
    all_labels = torch.cat(all_labels, 0).view(-1).numpy()
    probs = np.ones_like(all_labels) * p
    ap, weighted_ap, pr, roc = get_statistics(all_labels, probs)
    plot_pr(pr, "all_{}".format(p), "../figs/")

    all_labels = torch.FloatTensor(all_labels)
    probs = torch.FloatTensor(probs)
    loss = nn.functional.binary_cross_entropy(Variable(probs), Variable(all_labels)).data.view(-1)[0]
    print ("Loss {:.4f}".format(loss))

def main(args):
    model_name = args[1] # 'alexslicerZMIL'
    save_name = args[2]
    model_path = args[3] #"../models/alexslicerZMIL"
    lungs_dir = args[4] #"../input/3Darrays_visual/"
    labels_file = args[5] #"../input/stage1_solution_trim.csv"
    figs_dir = "../figs"

    print('-'*10)
    print("Model Name: {}".format(save_name))
    test_loader = get_lung_data(lungs_dir, labels_file)
    config = CONFIGS[model_name]
    model = config['net']()
    get_probs = config.get('get_probs', None)
    criterion = config.get('criterion', nn.functional.binary_cross_entropy)
    model.load_state_dict(torch.load(model_path))
    
    if torch.cuda.is_available():
        model = model.cuda()
    ap, mAP, pr_curve, roc_curve = evaluate_model(test_loader, model, criterion, get_probs)
    plot_pr(pr_curve, save_name, figs_dir)
    print('-'*10)

if __name__ == "__main__":
    #main(sys.argv)
    lungs_dir = "../input/arrays_notrim/"
    labels_file = "../input/stage1_solution_trim.csv"
    evaluate_dumb_model(1., lungs_dir, labels_file)
    evaluate_dumb_model(0.5, lungs_dir, labels_file)
    evaluate_dumb_model(0.25, lungs_dir, labels_file)
    evaluate_dumb_model(0., lungs_dir, labels_file)

