import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import copy
import argparse
import os

# Within package
import models
import util

def train_model(model,dset_loaders, criterion, optimizer, batch_size, lr_scheduler=None, num_epochs=25, verbose = False):
    since = time.time()

    best_model = model
    best_acc = 0.0
    trainlen = len(dset_loaders['train'])
    train_loss_history = []
    validation_loss_history = []

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
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
                    crit = nn.BCELoss()
                    loss = crit(outputs, labels)
                    if phase == 'train':
                        train_loss_history += [loss.data.cpu()]
                    else:
                        validation_loss_history += [loss.data.cpu()]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum((outputs.data > .5) == (labels.data > .5))
                    if phase == 'train' and verbose and i % 25 == 0:
                        print ("tr loss: {}".format(running_loss / (i + 1)))


                # Note: BCE loss already divides by batchsize
                epoch_loss = running_loss / (len(dset_loaders[phase]))
                epoch_acc = float(running_corrects) / (len(dset_loaders[phase]) * batch_size)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)

            flat_weights = []
            for param in model.parameters():
                flat_weights += [param.data.view(-1).cpu().numpy()]
            flat_weights = np.concatenate(flat_weights)
            plt.hist(flat_weights, 50)
            plt.savefig('../models/weights_hist_{}'.format(epoch))
    except KeyboardInterrupt:
        pass

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, train_loss_history, validation_loss_history


def main(data_path, labels_file, models_dir, save_name, load_name, train_net='3d'):
    batch_size = 4
    LR = 0.0001
    NUM_EPOCHS = 1
    WEIGHT_INIT = None
    optimizer_ft = None
    net = None

    if train_net == '3d':
        # cnn3d model
        net = models.Cnn3d(WEIGHT_INIT)
        data = util.get_data(data_path, labels_file, batch_size, crop=((0,60), (0,224), (0,224)), training_size=500)
        optimizer_ft = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.1)

    elif train_net == 'vgg3d':
        from vgg3d import get_pretrained_2D_layers
        net = get_pretrained_2D_layers()
        data = util.get_data(data_path, labels_file, batch_size, crop=((0,60), (0,224), (0,224)), training_size=20)
        optimizer_ft = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.1)

    elif train_net == 'simple':
        # alexnet model
        net = models.Simple()
        data = util.get_data(data_path, labels_file, batch_size,use_3d=False, crop=((30,33), (0,227), (0,227)))

    elif train_net == 'alex3d':
        # net alexnet model
        batch_size = 1 #everything is hard coded... whoops
        net = models.Alex3d()
        data = util.get_data(data_path, labels_file, batch_size, training_size = 500)
        optimizer_ft = torch.optim.Adam(net.predict.parameters(), lr=LR)
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.BCELoss()
    # Lung data is (60, 227 , 227), we want (60, 224, 224)
    #data = util.get_data(data_path, batch_size, crop=((30, 31), (0,224), (0,224)))
    if load_name != None:
        net.load_state_dict(torch.load(models_dir+load_name))
      
    plt.ioff()
    model_ft, train_loss, validation_loss = train_model(net, 
                                            data, 
                                            criterion,
                                            optimizer_ft,
                                            batch_size,
                                            num_epochs=NUM_EPOCHS,
                                            verbose=False)
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
    
    r = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: Cuda unavailable")
    main(r.DATA_DIR, r.LABELS_FILE, r.MODELS_DIR, r.SAVE_NAME, r.LOAD_MODEL, r.NET)
