import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import copy
import argparse

# Within package
import models
import util

def train_model(model,dset_loaders, criterion, optimizer, lr_scheduler=None, num_epochs=25, verbose = False):
    since = time.time()

    best_model = model
    best_acc = 0.0
    trainlen = len(dset_loaders['train'])

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
                weights = 0.75 * labels.data + 0.25 * (1 - labels.data)
                weights = weights.view(1,1).float()
                crit = nn.BCELoss(weight=weights)
                loss = crit(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum((outputs.data > .5) == (labels.data > .5))
                if phase == 'train' and verbose and i % 25 == 0:
                    print ("tr loss: {}".format(running_loss / (i + 1)))


            epoch_loss = running_loss / len(dset_loaders[phase])
            epoch_acc = float(running_corrects) / len(dset_loaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def main(data_path, labels_file):
    batch_size = 1
    LR = 0.0001
    NUM_EPOCHS = 30
    WEIGHT_INIT = 1e-3
    # cnn3d model
    # net = models.Cnn3d(WEIGHT_INIT)
    # data = util.get_data(data_path, labels_file, batch_size, crop=((0,60), (0,224), (0,224)))

    # alexnet model
    #net = models.Simple()
    #data = util.get_data(data_path, labels_file, batch_size,use_3d=False, crop=((30,33), (0,227), (0,227)))

    # net alexnet model
    net = models.Alex3d()
    data = util.get_data(data_path, labels_file, batch_size, training_size = 599)

    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.BCELoss()
    # Lung data is (60, 227 , 227), we want (60, 224, 224)
    #data = util.get_data(data_path, batch_size, crop=((30, 31), (0,224), (0,224)))
    optimizer_ft = torch.optim.Adam(net.predict.parameters(), lr=LR)
    model_ft = train_model(net, 
                           data, 
                           criterion,
                           optimizer_ft,
                           num_epochs=NUM_EPOCHS,
                           verbose=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/a/data/lungdl/balanced/', help="Path to data directory")
    parser.add_argument('--LABELS_FILE', default='/a/data/lungdl/balanced_shuffled_labels.csv', help="Path to data directory")
    r = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: Cuda unavailable")
    main(r.DATA_DIR, r.LABELS_FILE)
