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
import sklearn.metrics as metrics
from apmeter import APMeter

# Within package
import models
import util

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
                    #average_precision = metrics.average_precision_score(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
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

import vgg3d
import slicewise_models
CONFIGS = {
    'simple' : {
        'net': models.Simple,
        'crop' : ((30,33), (0,227), (0,227)),
    },
    '3d': {
        'net' : models.Cnn3d,
        'crop' : ((0,60),(0,224),(0,224)),
        'batch_size' : 3,
    },
    'alex3d' :{
        'net' : models.Alex3d,
        'crop' : ((0,60),(0,227),(0,227)),
        'params': 'predict',
        'batch_size' : 1, #ONLY WORKS WITH BATCHSIZE 1
        
    },
    'vgg3d' : {
        'net' : vgg3d.get_pretrained_2D_layers,
        'batch_size' : 3,
    },
    'alexslicer' :{
        'net' : slicewise_models.Alex,
        'params': 'predict',
        'lr': 0.00001,
        'lr_scheduler' : util.exp_lr_decay(0.00001, 0.85),
        'batch_size' : 20,
        'augment_data': True
    },
    'alexslicerMIL':{
        'net' : slicewise_models.AlexMIL,
        'params': 'mil_scores',
        'lr': 0.004, # overfits 20 examples with LR 0.004
        #'lr_scheduler' : util.exp_lr_decay(0.00001, 0.85),
        'batch_size' : 20,
        'augment_data': True,
        'criterion' : lambda y, t: util.sparse_BCE_loss(y, t, 0.00001),
        'get_probs' : lambda outs: outs[0],
    },
    'alexslicershallow' :{
        'net' : slicewise_models.AlexShallow,
        'params': 'predict',
        'lr': 0.000001,
        #'lr_scheduler' : util.exp_lr_decay(0.00005, 0.85),
        'batch_size' : 20,
        'augment_data': False
    },
    'alexslicershallowMIL':{
        'net' : slicewise_models.AlexShallowMIL,
        'params': 'mil_scores',
        'lr': 0.004, # overfits 20 examples with LR 0.004
        'lr_scheduler' : util.exp_lr_decay(0.004, 0.85),
        'batch_size' : 20,
        'augment_data': False,
        'criterion' : lambda y, t: util.sparse_BCE_loss(y, t, 0.00001),
        'get_probs' : lambda outs: outs[0],
    },
    'resnet50' : {
        'net' : lambda: slicewise_models.ResNet(50),
        'crop' : ((0,60),(0,225),(0,225)),
        'params': 'predict',
        'lr': 0.000005,
        'batch_size': 4,
        'lr_scheduler': util.exp_lr_decay(0.000005, 0.95),
        'augment_data': True
    },
    'resnet50MIL' : {
        'net' : lambda: slicewise_models.ResNetMIL(50),
        'crop' : ((0,60),(0,224),(0,224)),
        'params': 'mil_scores',
        'lr': 0.0005,
        'batch_size': 4,
        'lr_scheduler': util.exp_lr_decay(0.0005, 0.95),
        'augment_data': False,
        'criterion' : lambda y, t: util.sparse_BCE_loss(y, t, 0.00001),
        'get_probs' : lambda outs: outs[0],
    
    },
    'resnet152' : {
        'net' : lambda: slicewise_models.ResNet(152),
        'crop' : ((0,60),(0,225),(0,225)),
        'params': 'predict',
        'lr': 0.000005,
        'lr_scheduler': util.exp_lr_decay(0.000005, 0.95),
        'batch_size': 4,
        'augment_data': True,
        'reg': 0.0001
    },
    'resnet152boosted' : {
        'net': lambda: slicewise_models.ResNetBoosted(152),
        'crop' : ((0,60),(0,225),(0,225)),
        'xgboost': True,
        'batch_size': 4,
        'augment_data': False,
        'max_depth': 2
    },
    'alexboosted': {
        'net': lambda: slicewise_models.AlexBoosted(),
        'xgboost': True,
        'batch_size': 4,
        'augment_data': False,
        'max_depth': 3
    }
}
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

    data = util.get_data(data_path, labels_file, batch_size, crop=crop, 
                         training_size=training_size,
                         augment_data=augment_data)

    if xgboost:
        # can't train like a pytorch net
        net.train(data['train'], data['val'], NUM_EPOCHS, max_depth)

        return

    trainable_attr = config.get('params', None)
    params = None
    if trainable_attr is not None:
        params = getattr(net, trainable_attr).parameters()
    else:
        params = net.parameters()
        

    optimizer_ft = None

    optimizer_ft = torch.optim.Adam(params, lr=lr, weight_decay=reg)
    if torch.cuda.is_available():
        net = net.cuda()

    if load_name != None:
        net.load_state_dict(torch.load(models_dir+load_name))
      
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

    print("Saving net to disk at - {}".format(models_dir+save_name))
    torch.save(net.state_dict(), os.path.join(models_dir,save_name))
    torch.save(train_loss, os.path.join(models_dir,save_name + '_train_loss'))
    torch.save(validation_loss, os.path.join(models_dir,save_name + '_validation_loss'))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/a/data/lungdl/balanced/', help="Path to data directory")
    parser.add_argument('--LABELS_FILE', default='/a/data/lungdl/balanced_shuffled_labels.csv', help="Path to data directory")
    parser.add_argument('--NET', default='alex3d', help="One of: alex3d, 3d, simple")
    parser.add_argument('--MODELS_DIR', default='/a/data/lungdl/models/', help='Path to model directory')
    parser.add_argument('--SAVE_NAME', default='tmp.model', help='Name of save model')
    parser.add_argument('--LOAD_MODEL', default=None, help='Load pretrained model')
    parser.add_argument('--NUM_EPOCHS',  '-n', type=int, default=20, help='number of epochs to run')
    parser.add_argument('--TRAINING_SIZE',  '-s', type=int, default=500, help='number of')
    parser.add_argument('--NO_VAL', type=bool, nargs='?', const=True, default=False, help="Don't perform validation step")
    
    r = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: Cuda unavailable")
    main(r)
