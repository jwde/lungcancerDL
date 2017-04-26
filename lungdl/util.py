"""
util

util provides functions for manipulating data before processing by the network

"""
import torch
import torch.utils.data as data
import csv
import numpy as np
from itertools import islice

def hu_to_visual_features(img, low, high):
    """
    Parameters:
        img := 3D array of hu units
        low := everything less than low gets maped to low
        high := everything more than high gets mapped to high
    """

    # Make a deep copy to avoid np pointer craziness...
    # TODO: does this need to happen?
    new_image = np.copy(img)

    # Threshold the values
    new_image[new_image < low] = low
    new_image[new_image > high] = high

    # Scale the values
    new_image -= low
    new_image = new_image / float(high - low)

    return new_image

def load_img(path):
    return np.load(path)

class LabeledKaggleDataset(data.Dataset):
    def __init__(self, image_dir, labels_path, slice_start, slice_end, input_transform=None, target_transform=None):
        super(LabeledKaggleDataset, self).__init__()
        self.image_dir = image_dir
        self.lung_names = []
        self.lung_labels = []
        self.input_transform = input_transform
        self.target_transform = target_transform
        with open(labels_path) as csv_labels:
            labels_reader = csv.reader(csv_labels)
            for row in islice(labels_reader, 1 + slice_start, 1 + slice_end):
                self.lung_names += [row[0]]
                self.lung_labels += [int(row[1])]

    def __getitem__(self, index):
        f = self.lung_names[index] + '.npy'
        img = load_img(self.image_dir + f)
        #img = hu_to_visual_features(img, -1500, 500)
        # Uncommented on Jason Branch - I dont have the pre thresholding data
        img = torch.from_numpy(img).float()
        target = torch.FloatTensor(1)
        target[0] = self.lung_labels[index]
        if self.input_transform:
            img = self.input_transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        img = img.view(1,60, 227, 227)
        img = img[:,:,:224,:224]
        return img, target

    def __len__(self):
        return len(self.lung_names)
            

def get_training_set(lungs_dir, labels_file):
    return LabeledKaggleDataset(lungs_dir, labels_file, 0, 50)

def get_test_set(lungs_dir, labels_file):
    return LabeledKaggleDataset(lungs_dir, labels_file, 50, 60)

class LabeledKaggleDataset2D(LabeledKaggleDataset):
    def __init__(self, image_dir, labels_path, slice_start, slice_end, input_transform=None, target_transform=None):
        super(LabeledKaggleDataset2D, self).__init__(image_dir, labels_path, slice_start, slice_end, input_transform, target_transform)

    def __getitem__(self, index):
        f = self.lung_names[index] + '.npy'
        img = load_img(self.image_dir + f)
        img = hu_to_visual_features(img, -1500, 500)
        img = torch.from_numpy(img)
        target = torch.doubletensor(1, 1)
        target[0,0] = self.lung_labels[index]
        img = img[30]
        G = img.clone()
        B = img.clone()
        img3c = torch.stack((img, G, B))
        if self.input_transform:
            img3c = self.input_transform(imgc)
        if self.target_transform:
            target = self.target_transform(target)
        

        return img3c, target

def get_training_set2D(lungs_dir, labels_file, input_transfoj=None):
    return LabeledKaggleDataset2D(lungs_dir, labels_file, 0, 100, input_transform)

def get_test_set2D(lungs_dir, labels_file, input_transform=None):
    return LabeledKaggleDataset2D(lungs_dir, labels_file, 100, 120, input_transform)

def get_data(data_path, batch_size, use_3d = True):
    if not use_3d:
        print("Only 3d data loading is supported at this time")
    lungs_dir = data_path + '3Darrays_visual/'
    labels_file = data_path + 'stage1_labels.csv'
    trainset = get_training_set(lungs_dir, labels_file)
    testset = get_test_set(lungs_dir, labels_file)
    # Parallel loader breaks on the aws machine python2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)#, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)#, num_workers=1)
    return { "train" : trainloader, "val" : testloader}

def rand_interval(low, high):
    return (low-high) * np.random.random_sample() + low

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
