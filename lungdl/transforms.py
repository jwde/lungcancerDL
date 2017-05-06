import torch
import numpy as np
import scipy.ndimage
import random

class ToTensor(object):
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            print ("Calling To Tensor on Non numpy array")
        im = torch.from_numpy(img)
        return im.float()

class Shift(object):
    def __init__(self, shift):
        self.shift = shift
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            print ("Calling To Tensor on Non numpy array")
        return scipy.ndimage.shift(img, self.shift, order = 0)

class Flip(object):
    """Given C x H x W numpy, flips"""
    def __init__(self, horizontal = True):
        self.horizontal = horizontal
    def __call__(self, img):
        if self.horizontal:
            return img[:,:,::-1].copy()
        return img[:,::-1,:].copy()

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            # copy is necessary as torch does not like negative indexing
            return img[:,:,::-1].copy() 
        return img

class RandomShift(object):
    """ shifts an image somewhere between -max and max for each dimension"""
    def __init__(self, max_shift):
        self.max_shift = max_shift

    def __call__(self, img):
        shift = self.max_shift * np.random.uniform(-1,1,3) # 3 dims C,H,W
        return scipy.ndimage.shift(img, shift, order = 0)

class RandomRotation(object):
    def __init__(self, rot):
        self.max_rot = rot

    def __call__(self, img):
        rot = (2*random.random() - 1) * self.max_rot
        return scipy.ndimage.interpolation.rotate(img, rot, (2, 1), reshape = False, order = 0)

