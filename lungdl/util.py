"""
util

util provides functions for manipulating data before processing by the network

"""
import torch
import torch.utils.data as data
import csv
import numpy as np
import multiprocessing
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
    def __init__(self, image_dir, labels_path, slice_start = None, slice_end = None, use_3d = True, crop = None,
                 input_transform=None, target_transform=None):
        super(LabeledKaggleDataset, self).__init__()
        self.image_dir = image_dir
        self.lung_names = []
        self.lung_labels = []
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.crop = crop
        self.use_3d = use_3d
        with open(labels_path) as csv_labels:
            labels_reader = csv.reader(csv_labels)
            next(labels_reader) # skip header
            for row in islice(labels_reader, slice_start, slice_end):
            #for row in islice(labels_reader, 1 + slice_start, 1 + slice_end):
                self.lung_names += [row[0]]
                self.lung_labels += [int(row[1])]

    def __getitem__(self, index):
        f = self.lung_names[index] + '.npy'
        img = load_img(self.image_dir + f)
        img = hu_to_visual_features(img, -1500, 500)
        # Uncommented on Jason Branch - I dont have the pre thresholding data
        img = torch.from_numpy(img).float()
        target = torch.FloatTensor(1)
        target[0] = self.lung_labels[index]
        if self.input_transform:
            img = self.input_transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        img = img.view(-1,60, 227, 227)
        crop = self.crop
        if crop:
            img = img[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
        if not self.use_3d:
            # If not use 3d, we overload the Channels to encode depth
            size = img.size()
            img = img.view (size[1], size[2], size[3])

        return img, target

    def __len__(self):
        return len(self.lung_names)

class LabeledKaggleRamDataset(data.Dataset):
    def __init__(self, image_dir, labels_path, slice_start = None, slice_end = None, use_3d = True, crop = None,
                 input_transform=None, target_transform=None):
        super(LabeledKaggleRamDataset, self).__init__()
        self.slow_dataset = LabeledKaggleDataset(image_dir, labels_path, slice_start, slice_end, use_3d, crop,
                 input_transform, target_transform)
        self.images = []
        self.targets = []
        for i in range(len(self.slow_dataset)):
            img, target = self.slow_dataset[i]
            self.images += [img]
            self.targets += [target]

    def __len__(self):
        return len(self.slow_dataset)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

def get_data(lungs_dir, labels_file, batch_size, use_3d = True, crop = None, training_size = 600):
    num_cores = multiprocessing.cpu_count()
    trainset = LabeledKaggleRamDataset(lungs_dir, labels_file, None, training_size, use_3d = use_3d, crop = crop)
    testset = LabeledKaggleRamDataset(lungs_dir, labels_file,training_size, None, use_3d = use_3d, crop = crop)
    # Parallel loader breaks on the aws machine python2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_cores)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_cores)
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
