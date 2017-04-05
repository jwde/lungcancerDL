import csv
import os
from itertools import islice
import numpy as np

DATA_DIR = '/notebooks/sharedfolder/lungcancerdl/input/'
lungs_dir = DATA_DIR + '3Darrays_visual/'
labels_file = DATA_DIR + 'stage1_labels.csv'
vgg_features_dir = DATA_DIR + 'vgg_training/'
vgg19_features_dir = DATA_DIR + 'vgg19_training/'

def get_labels_by_name():
    if not hasattr(get_labels_by_name, 'labels_by_name'):
        get_labels_by_name.labels_by_name = {}
        with open(labels_file, 'rb') as csv_labels:
            labels_reader = csv.reader(csv_labels)
            for row in islice(labels_reader, 1, None):
                if not row[0] == '059d8c14b2256a2ba4e38ac511700203' and not row[0] == '66b7666912e1d469cd9817a1cade694c':
                    get_labels_by_name.labels_by_name[row[0]] = int(row[1])
    return get_labels_by_name.labels_by_name

def get_training_lungs():
    for lung_id in get_labels_by_name().keys():
        f = lung_id + '.npy'
        lung_img = np.load(lungs_dir + f)
        yield lung_img[:, 1:-2, 1:-2], lung_id

def get_training_lung_labels():
    for lung_id in get_labels_by_name().keys():
        yield get_labels_by_name()[lung_id]

def get_training_slice_labels():
    for lung_img, lung_id in get_training_lungs():
        for slice_num in xrange(lung_img.shape[0]):
            yield get_labels_by_name()[lung_id]

def get_training_slices():
    for lung_img, lung_id in get_training_lungs():
        for ct_slice in lung_img:
            yield ct_slice, lung_id

def get_test_lungs():
    for _, _, files in os.walk(lungs_dir):
        for f in files:
            lung_id = f.split('.')[0]
            if not lung_id in get_labels_by_name():
                lung_img = np.load(lungs_dir + f)
                yield lung_img

def get_training_vgg_features():
    for lung_id in get_labels_by_name().keys():
        f = lung_id + '.npy'
        lung_features = np.load(vgg_features_dir + f)
        yield lung_features, lung_id

def get_training_vgg19_features():
    for lung_id in get_labels_by_name().keys():
        f = lung_id + '.npy'
        lung_features = np.load(vgg19_features_dir + f)
        yield lung_features, lung_id
