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
