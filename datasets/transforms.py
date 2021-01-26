import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, normals):
        for t in self.transforms:
            image, normals = t(image, normals)
        return image, normals


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, normals):
        image = F.center_crop(image, self.size)
        normals = F.center_crop(normals, self.size)
        return image, normals

class Resize(object): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, normals): #TODO warning for mask -> nearest interpolation?
        image = F.resize(image, self.size, self.interpolation)
        normals = F.resize(normals, self.size, self.interpolation)
        return image, normals


class ToTensor(object):
    def __call__(self, image, normals):
        image = F.to_tensor(image)
        normals = F.to_tensor(normals)
        #normals = torch.as_tensor(np.array(normals), dtype=torch.int64) #for masks
        return image, normals


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, normals):
        image = F.normalize(image, mean=self.mean, std=self.std)
        normals = F.normalize(normals, mean=self.mean, std=self.std)
        return image, normals


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            normals = F.hflip(normals) #TODO change normals
        return image, normals

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            normals = F.vflip(normals) 
            #TODO change normals
        return image, normals


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, normals):
        image = pad_if_smaller(image, self.size)
        normals = pad_if_smaller(normals, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        normals = F.crop(normals, *crop_params)
        return image, normals



class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, image, normals): #TODO warning for mask -> nearest interpolation?
        size = np.random.randint(self.low, self.high)
        image = F.resize(image, size, self.interpolation)
        normals = F.resize(normals, size, self.interpolation)
        return image, normals

class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = degrees

    def __call__(self, image, normals): #TODO warning for mask -> nearest interpolation?
        angle = T.RandomRotation.get_params(self.degrees)
        image =  F.rotate(image, angle, self.resample, self.expand, self.center, self.fill)
        normals =  F.rotate(normals, angle, self.resample, self.expand, self.center, self.fill) 
        #TODO change normals
        return image, normals
