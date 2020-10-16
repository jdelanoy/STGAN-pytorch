import os
import math
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import random
import numpy as np

def make_dataset(root, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset.txt'), 'r')]
    all_attr_names = lines[0].split()
    print(all_attr_names)
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    np.random.seed(10)
    lines = lines[1:]
    if mode == 'train':
        lines = lines[985:]  # train set contains 200599 images 985:
    if mode == 'val':
        np.random.shuffle(lines)
        lines = lines[:]  # val set contains 200 images :992
    if mode == 'test':
        # #only from havran
        # lines = lines[:985]  # test set contains 1800 images
        #only from one shape/one env
        shape=""
        env=""
        lines=[line for line in lines if (shape in line and env in line)]
        # #all

        #take 100 random images
        np.random.shuffle(lines)
        lines=lines[:200]
    #print(len(lines))
    items = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(float(values[idx])*2-1)
        items.append([filename, label])
        #print([filename, label])
    return items


class MaterialDataset(data.Dataset):
    def __init__(self, root, mode, selected_attrs, transform=None):
        self.items = make_dataset(root, mode, selected_attrs)
        self.root = root
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, '256px_dataset', filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)


class MaterialDataLoader(object):
    def __init__(self, root, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16):
        if mode not in ['train', 'test',]:
            return

        transform = []
        if crop_size is not None:
            transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if mode == 'train':
            val_transform = transforms.Compose(transform)       # make val loader before transform is inserted
            val_set = MaterialDataset(root, 'val', selected_attrs, transform=val_transform)
            self.val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            transform.insert(0, transforms.RandomHorizontalFlip())
            train_transform = transforms.Compose(transform)
            train_set = MaterialDataset(root, 'train', selected_attrs, transform=train_transform)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_transform = transforms.Compose(transform)
            test_set = MaterialDataset(root, 'test', selected_attrs, transform=test_transform)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
