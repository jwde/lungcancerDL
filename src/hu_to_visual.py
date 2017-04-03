import numpy as np
from lung_utils import *
import os

from joblib import Parallel, delayed
import multiprocessing



INPUT_DIR = '/tmp/input/3Darrays_stage1_2/'
OUTPUT_DIR = '/tmp/input/3Darrays_visual/'

SAMPLE = None
#SAMPLE = 20
HU_MIN = -1500
HU_MAX = 500

# Setup patient list...
patients = os.listdir(INPUT_DIR)
patients.sort()
# for some odd reason there are duplicate files for each folder. Each duplicate starts with '._'

npat = len(patients)
for i in range(npat):
    if '._' in patients[i]:
        patients[i] = []
patients = list(filter(None,patients))
if SAMPLE != None:
    patients = patients[:SAMPLE]

npat = len(patients)

print (len(patients))


def main_loop(in_dir, out_dir , patient):
    lung = np.load(in_dir + patient)
    new_lung = hu_to_visual_features(lung, HU_MIN, HU_MAX)
    outpath = out_dir + patient
    np.save(outpath, new_lung)


#main_loop(INPUT_DIR, OUTPUT_DIR, SAMPLE)
num_cores = 8

iterations = range(npat)

# Run parallelized loop over all patients
Parallel(n_jobs=num_cores)(delayed(main_loop)(INPUT_DIR, OUTPUT_DIR,patients[i]) for i in iterations)
