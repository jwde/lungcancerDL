#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import copy
import argparse
import os
from torchvision import datasets, models, transforms
import sklearn.metrics as metrics
from apmeter import APMeter

# Within package
import models
import mil2d_models
import util
from ants_and_bees_data import get_ants_and_bees
from config import CONFIGS

def train_model(model,dset_loaders, criterion, optimizer, batch_size,
                get_probs = None ,lr_scheduler=None, num_epochs=25,
                verbose = False, validation=True):
    since = time.time()

    best_model = model
    best_acc = 0.0
    trainlen = len(dset_loaders['train'])
    train_loss_history = []
    validation_loss_history = []

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            phases = ['train', 'val'] if validation else ['train']

            for phase in phases:
                ap_total = {p:APMeter() for p in phases}
                if phase == 'train':
                    if lr_scheduler is not None:
                        optimizer = lr_scheduler(optimizer, epoch)
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, data in enumerate(dset_loaders[phase]):
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda()), \
                            Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    labels = labels.float()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs.data, 1)
                    #print (labels.size())
                    #labels have size (batch, 1,1)?
                    #weights = 0.75 * labels.data + 0.25 * (1 - labels.data)
                    #weights = weights.view(1,1).float()
                    #crit = nn.BCELoss(weight=weights)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        train_loss_history += [loss.data.cpu()]
                    else:
                        validation_loss_history += [loss.data.cpu()]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    if get_probs:
                        outputs = get_probs(outputs)
                    average_precision = metrics.average_precision_score(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
                    ap_total[phase].add(outputs.data, labels.data)
                    running_loss += loss.data[0]
                    running_corrects += torch.sum((outputs.data > .5) == (labels.data > .5))
                    if phase == 'train' and verbose and i % 25 == 0:
                        print ("tr loss: {}".format(running_loss / (i + 1)))


                # Note: BCE loss already divides by batchsize
                epoch_loss = running_loss / (len(dset_loaders[phase]))
                epoch_acc = float(running_corrects) / (len(dset_loaders[phase]) * batch_size)

                print('{} Loss: {:.4f} Acc: {:.4f} AP {:.4f}'.format(phase, epoch_loss, epoch_acc, ap_total[phase].value()[0]))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)

            #flat_weights = []
            #for param in model.parameters():
            #    flat_weights += [param.data.view(-1).cpu().numpy()]
            #flat_weights = np.concatenate(flat_weights)
            #plt.hist(flat_weights, 50)
            #plt.savefig('../models/weights_hist_{}'.format(epoch))

            time_elapsed = time.time() - since
            print('Time spent so far: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    except KeyboardInterrupt:
        pass

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, train_loss_history, validation_loss_history

# Functions to parse configurations
def get_dset_loader(dataset, dset_config):
    dset_loaders = None
    if dataset == 'ants_and_bees':
        dset_loaders = get_ants_and_bees(dset_config)
    else:
        print("ERROR: {} dataset is not defined".format(dataset))
        raise Exception

    return dset_loaders

def get_model(model):
    return {
        'alex': mil2d_models.alexnet256,
        'alexMIL': mil2d_models.alexnetMIL
    }[model]()

def get_criterion(criterion):
    return {
        'bce': nn.functional.binary_cross_entropy,
        'bce_sparse': lambda o,l: util.sparse_BCE_loss(o,l,reg=0)
    }[criterion['type']]

def main(r):
    # Disable interactive mode for matplotlib so docker wont segfault
    #plt.ioff()

    use_validation = not r.NO_VAL

    test = r.TEST
    config = CONFIGS[test]

    print("Training Test: \n")
    print(config['info'])
    print('-'*10)

    #models_dir = config.get('models_dir',"/a/data/lungdl/models")
    models_dir = r.MODELS_DIR

    net = get_model(config['model'])
    NUM_EPOCHS = config['epochs']
    reg = config.get('reg', 0.)
    batch_size = config.get('batch_size', 1)

    lr = config['init_lr']
    decay = config.get('decay', None)
    lr_scheduler = None
    if decay:
        lr_scheduler = util.exp_lr_decay(lr, decay)

    criterion = get_criterion(config['criterion'])
    is_tuple = config.get('is_tuple', None)


    #data = util.get_data(data_path, labels_file, batch_size, crop=crop,
    #                     training_size=training_size,
    #                     augment_data=augment_data)
    dset_config = config.get('dset_config', None)
    data = get_dset_loader(config['dataset'], dset_config)

    # If XGBOOST, train XGBOOST
    xgboost = config.get('xgboost', False)
    if xgboost:
        # can't train like a pytorch net

        max_depth = config.get('max_depth', 10)
        net.train(data['train'], data['val'], NUM_EPOCHS, max_depth)

        return

    # Get trainable parameters by attribute name (string)
    trainable_attr = config.get('params', None)
    params = None
    if trainable_attr is not None:
        params = getattr(net, trainable_attr).parameters()
    else:
        params = net.parameters()

    if torch.cuda.is_available():
        net = net.cuda()

    load_name = config.get('load_model', None)
    if load_name != None:
        net.load_state_dict(torch.load(models_dir+load_name))

    # How to get probabilities
    get_probs = None
    if is_tuple:
        get_probs = lambda x: x[0]
    else:
        get_probs = lambda x: x

    optimizer_ft = torch.optim.Adam(params, lr=lr, weight_decay=reg)
    model_ft, train_loss, validation_loss = train_model(net,
                                            data,
                                            criterion,
                                            optimizer_ft,
                                            batch_size,
                                            get_probs=get_probs,
                                            lr_scheduler=lr_scheduler,
                                            num_epochs=NUM_EPOCHS,
                                            verbose=False,
                                            validation = use_validation)

    model_path = os.path.join(models_dir, test)
    print("Saving net to disk at - {}".format(model_path))
    # TODO: abstract away the saving, include precision vs recall info
    torch.save(net.state_dict(), model_path + '.model')
    torch.save(train_loss, model_path +  '_train_loss')
    torch.save(validation_loss, model_path + '_validation_loss')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TEST', '-t', default='ants_bees', help="One of: alex3d, 3d, simple")
    parser.add_argument('--LOAD_MODEL', default=None, help='Load pretrained model')
    parser.add_argument('--NO_VAL', type=bool, nargs='?', const=True, default=False, help="Don't perform validation step")
    parser.add_argument('--MODELS_DIR', default='../models/')

    r = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: Cuda unavailable")
    main(r)
