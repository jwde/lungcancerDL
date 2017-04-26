import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import time
import copy

from model3d import Cnn3d
from data import get_training_set, get_test_set

#CONFIGS : TODO - move this to arg parser
DATA_DIR = '../input/'
lungs_dir = DATA_DIR + '3Darrays_visual/'
labels_file = DATA_DIR + 'stage1_labels.csv'
batch_size = 1


#### HELPER FUNCTIUNS

def get_dset_loaders(lungs_dir, labels_file):
    trainset = get_training_set(lungs_dir, labels_file)
    testset = get_test_set(lungs_dir, labels_file)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return { "train" : trainloader, "val" : testloader}

def train_model(model,dset_loaders, criterion, optimizer, lr_scheduler=None, num_epochs=25, verbose = False,
    cancer_weight=0.75):
    since = time.time()

    best_model = model
    best_acc = 0.0
    best_loss = 0.0
    best_trloss = 0.0
    best_tracc = 0.0
    trainlen = len(dset_loaders['train'])

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

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
                weights = cancer_weight * labels.data + (1.0 - cancer_weight) * (1 - labels.data)
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
            epoch_acc = running_corrects / len(dset_loaders[phase])

            #print_stats(phase, epoch_loss, epoch_acc)
            if phase == 'train' and epoch_acc > best_tracc:
                best_tracc = epoch_acc
                best_trloss = epoch_loss

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)


    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    stats = 'tr_l: {:.4f}, val_l: {:.4f}, tr_acc: {:.4f}, val_acc: {:.4f}    -    {:.0f}m {:.0f}s'.format(
        best_trloss, best_loss, best_tracc, best_acc, time_elapsed // 60, time_elapsed % 60)
    print(stats)
    return best_model, stats

def print_stats(phase, loss, acc):
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, loss, acc))

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

####
LR = 0.0001
MOMENTUM = 0.9
NUM_EPOCHS = 8
WEIGHT_INIT = 1e-3
DECAY = None
def randval(low, high):
    return (high - low) * np.random.random_sample() + low
def main():
    print('-' * 10)
    LR = randval( 0.00001, 0.01,)
    WEIGHT_INIT = randval(0.00001, 0.01)
    cancer_weight = randval(0.7, 0.9)
    MOMENTUM = randval(0.7, 0.95)
    print ("Es: {}, LR : {}, weight : {}, cancer_weight:, {}".format(NUM_EPOCHS, LR, WEIGHT_INIT, cancer_weight))
    net = Cnn3d(WEIGHT_INIT)
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.BCELoss()
    dset_loaders = get_dset_loaders(lungs_dir, labels_file)
    optimizer_ft = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    #optimizer_ft = optim.Adam(net.parameters(), lr=LR)
    model_ft, stats = train_model(net,dset_loaders, criterion, optimizer_ft, num_epochs=NUM_EPOCHS, verbose=False, cancer_weight=cancer_weight)

if __name__ == '__main__':
    main()