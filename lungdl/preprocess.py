import numpy  as np
import os
DATA_DIR = '/a/data/lungdl/'
INPUT_FOLDER = DATA_DIR + '3D_pruned/'
labels = DATA_DIR + 'stage1_labels.csv'

files = os.listdir(INPUT_FOLDER)

good_set = set()
for f in files:
    good_set.add((f.split('.')[0]))

label_data = np.loadtxt(labels, delimiter=',', skiprows = 1, dtype={'names' : ('id', 'label'), 'formats': ('S50', 'i')})
label_dict = {name: label for name, label in label_data}
good_label_dict = {name: label for name, label in label_data if name in good_set}
good_labels = [(name, label) for name, label in label_data if name in good_set]

count = 0
total = 0
for name, label in good_labels:
    count += label
    total += 1

#print count

#print "true percent = ", float(count)/total
#print (files)
good_set = set()
for f in files:
    good_set.add((f.split('.')[0]))

def print_to_file(labels):

    print ('id,cancer')
    for name, label in labels:
        print(name + ',' + str(label))

#print_to_file(good_labels)


zero_count = 0
half_and_half = []

#print (count)
for name, label in good_labels:
    if label == 0:
        if zero_count >= count:
            continue
        zero_count += 1
        half_and_half.append((name, label))
    else:
        half_and_half.append((name, label))
#print (len(half_and_half))

#import shutil
#i = 0
#for name, label in half_and_half:
#    print i
#    i += 1
#    shutil.copy(INPUT_FOLDER + name + '.npy', DATA_DIR + 'balanced/' + name + '.npy')
#
#print_to_file(DATA_DIR + 'balanced/', 
#

import random
random.shuffle(half_and_half)
print_to_file(half_and_half)

