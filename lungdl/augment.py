import util
import argparse
import os
import time
import explore
import scipy
import numpy as np
import torchvision.transforms as tt
import transforms


def flip_lr(img):
    transform = tt.Scale(2)
    print (img.numpy().shape)
    return transform

def test_load_time(args):
    since = time.time()
    data = util.get_data(args.INPUT_FOLDER, args.LABELS_FILE, 10)
    for i,t in enumerate(data["train"]):
        pass
    for i,t in enumerate(data["train"]):
        pass
    time_elapsed = time.time() - since
    print("Done, time_elapsed =", time_elapsed)

def main(args):
    if not os.path.exists(args.OUT_FOLDER):
        os.makedirs(args.OUT_FOLDER)
    since = time.time()

    transform = tt.Compose([transforms.RandomShift((10,50, 50)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
    data = util.LabeledKaggleDataset(args.INPUT_FOLDER, args.LABELS_FILE,
                input_transform = transform)
    
#    for i,(img,t) in enumerate(data):
    img, t = data.__getitem__(0)
    img = img.numpy()
    print (img.shape)
    explore.plot(img)
    #img_new = horizontal_shift(img)
    #explore.plot(img_new)
    time_elapsed = time.time() - since
    print("Done, time_elapsed =", time_elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_FOLDER', help="Path to input data directory")
    parser.add_argument('LABELS_FILE', help="Path to input data directory")
    parser.add_argument('OUT_FOLDER', help="Path to output data directory")
    r = parser.parse_args()
    main(r)
