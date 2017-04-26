import torch
import torch.utils.data as data
import csv
import numpy as np
from itertools import islice
from .lung_utils import hu_to_visual_features

#DATA_DIR = '/notebooks/sharedfolder/lungcancerdl/input/'
#lungs_dir = DATA_DIR + '3Darrays_visual/'
#labels_file = DATA_DIR + 'stage1_labels.csv'

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
        img = hu_to_visual_features(img, -1500, 500)
        img = torch.from_numpy(img).float()
        target = torch.FloatTensor(1, 1)
        target[0,0] = self.lung_labels[index]
        if self.input_transform:
            img = self.input_transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        img = img.view(1,60, 227, 227)
        return img, target

    def __len__(self):
        return len(self.lung_names)
            

def get_training_set(lungs_dir, labels_file):
    return LabeledKaggleDataset(lungs_dir, labels_file, 0, 100)

def get_test_set(lungs_dir, labels_file):
    #return LabeledKaggleDataset(lungs_dir, labels_file, 1350, 1397)
    return LabeledKaggleDataset(lungs_dir, labels_file, 150, 200)

class LabeledKaggleDataset2D(LabeledKaggleDataset):
    def __init__(self, image_dir, labels_path, slice_start, slice_end, input_transform=None, target_transform=None):
        super(LabeledKaggleDataset2D, self).__init__(image_dir, labels_path, slice_start, slice_end, input_transform, target_transform)

    def __getitem__(self, index):
        f = self.lung_names[index] + '.npy'
        img = load_img(self.image_dir + f)
        img = hu_to_visual_features(img, -1500, 500)
        img = torch.from_numpy(img).float()
        target = torch.FloatTensor(1, 1)
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
    return LabeledKaggleDataset2D(lungs_dir, labels_file, 0, 1000, input_transform)

def get_test_set2D(lungs_dir, labels_file, input_transform=None):
    return LabeledKaggleDataset2D(lungs_dir, labels_file, 1000, 1397, input_transform)


