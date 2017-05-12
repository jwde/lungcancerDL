from torchvision import transforms, datasets
import torch

import os
import random


# CxWxH --> C x Z x W x H
def to_3d(z = 60):
    def all_else_black(img2d):
        img2d = img2d.mean(0) # convert to black and white
        C, W, H = img2d.size()
        w = int(random.random() * (227 - W))
        h = int(random.random() * (227 - H))
        W_new = 227
        H_new = 227
        new = torch.FloatTensor(C, z, W_new, H_new).zero_()
        sl = int(random.random() * z)
        new[:, sl, w:w+W, h:h+H] = img2d
        return new
    return all_else_black

def get_transforms(dset_config):
    data_transforms = None
    if dset_config is None:
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomSizedCrop(227),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    to_bw
                ]),
                'val': transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(227),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    to_bw
                ]),
            }
    else:
        img_size = 227
        embed_size = dset_config.get('size', 227)
        peturb_xy = dset_config.get('peturb_xy', False)
        data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomSizedCrop(img_size),
                    transforms.Scale(embed_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_embed(img_size, peturb_xy),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    to_bw
                ]),
                'val': transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(img_size),
                    transforms.Scale(embed_size),
                    transforms.ToTensor(),
                    random_embed(img_size, peturb_xy),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    to_bw
                ]),
            }

    return data_transforms

def to_bw(img2d):
    return img2d.mean(0)

def random_embed(size, peturb_xy=False):
    def embed(img):
        C,W,H = img.size()
        x = 0
        y = 0
        if peturb_xy:
            x = int(random.random() * (size - W))
            y = int(random.random() * (size - H))
        new = torch.FloatTensor(C, size, size).zero_()
        new[:, x: x + W, y: y + H] = img
        return new
    return embed




def get_ants_and_bees(dset_config):
    data_transforms = get_transforms(dset_config)
    #data_dir = '/a/data/lungdl/hymenoptera_data'
    #data_dir = '../input/hymenoptera_data'
    data_dir = 'input/hymenoptera_data'
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    return dset_loaders

