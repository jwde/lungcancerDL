import csv
import os
from itertools import islice

lungs_dir = '../input/3Darrays_stage1_2/'
labels_file = '../input/stage1_labels.csv'

def get_labels_by_name():
    if not hasattr(get_labels_by_name, 'labels_by_name'):
        get_labels_by_name.labels_by_name = {}
        with open(labels_file, 'rb') as csv_labels:
            labels_reader = csv.reader(csv_labels)
            for row in islice(labels_reader, 1, None):
                get_labels_by_name.labels_by_name[row[0]] = int(row[1])
    return get_labels_by_name.labels_by_name

def get_training_lungs():
    for lung_id in get_labels_by_name().keys():
        f = lung_id + '.npy'
        lung_img = np.load(lungs_dir + f)
        yield lung_img

def get_training_lung_labels():
    for lung_id in get_labels_by_name().keys():
        yield get_labels_by_name()[lung_id]

def get_training_slice_labels():
    for lung_img in get_training_lungs():
        for slice_num in xrange(lung_img.shape[0]):
            yield get_labels_by_name()[lung_id]

def get_training_slices():
    for lung_img in get_training_lungs():
        for ct_slice in lung_img:
            yield ct_slice.transpose(1, 2, 0)[1:-2,1:-2,:]

def get_test_lungs():
    for _, _, files in os.walk(lungs_dir):
        for f in files:
            lung_id = f.split('.')[0]
            if not lung_id in get_labels_by_name():
                lung_img = np.load(lungs_dir + f)
                yield lung_img
