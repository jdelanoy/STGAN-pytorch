import os
import math
import torch
from torch.utils import data
from torchvision import transforms as Tvision
import datasets.transforms2 as T
from PIL import Image
import random
import numpy as np
#from utils.im_util import get_alpha_channel
import cv2
from matplotlib import pyplot as plt


def make_dataset(root, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines_train = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_train.txt'), 'r')]
    lines_test = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_test_full.txt'), 'r')]
    all_attr_names = lines_train[0].split()
    print(mode,all_attr_names, len(lines_train))
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    np.random.seed(10)
    random.seed(18)
    lines_train=lines_train[1:]
    lines_test=lines_test[1:]
    #lines_train = lines_train[1:]
    if mode == 'train':
        lines = lines_train
    if mode == 'val': #put in first half a batch of test images, half of training images
        lines = random.sample(lines_test,16)+random.sample(lines_train,16)
    if mode == 'test':
        #np.random.shuffle(lines_test)
        lines = lines_test #[:32*2]+random.sample(lines_train,32*4) #for full dataset

    files = []
    mat_attrs = []

    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        mat_attr = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            mat_attr.append(float(values[idx]) * 2 - 1)

        files.append(filename)
        mat_attrs.append(mat_attr)
        filename_split = filename.split('/')[-1].split('@')[-1].split('.')[0].split('-')


    return {'files': files,
            'mat_attrs': mat_attrs}



class MaterialDataset(data.Dataset):
    def __init__(self, root, mode, selected_attrs=[], disentangled=False, transform=None, mask_input_bg=True):
        items = make_dataset(root, mode, selected_attrs)

        self.files = items['files']
        self.mat_attrs = items['mat_attrs']


        self.root = root
        self.mode = mode
        self.transform = transform
        self.mask_input_bg = mask_input_bg

    def __getitem__(self, index_and_mode):
        index = index_and_mode
        
        mat_attr = self.mat_attrs[index]

        ##### OpenCV version
        image_rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, "renderings", self.files[index]), 1), cv2.COLOR_BGR2RGB)
        size=image_rgb.shape[0]
        #read normals and the mask
        normals_bgra = cv2.imread(os.path.join(self.root, "normals", self.files[index][:-3]+"png"), -1)
        if normals_bgra.shape[0] != size :
            normals_bgra = cv2.resize(normals_bgra,(size,size))
        normals = np.ndarray((size,size,4), dtype=np.uint8)
        cv2.mixChannels([normals_bgra], [normals], [0,2, 1,1, 2,0, 3,3]) #BGRA -> RGBA

        #transfer alpha channel to the image
        image = np.ndarray(normals.shape, dtype=np.uint8)
        cv2.mixChannels([image_rgb,normals], [image], [0,0, 1,1, 2,2, 6,3])
        
        #normals.putalpha(mask)
        if self.transform is not None:
            image,normals = self.transform(image,normals) 
        
        if self.mask_input_bg:
            image = image*image[3:]
            normals = normals*normals[3:]

        return image,normals

    def __len__(self):
        return len(self.files)



class MaterialDataLoader(object):
    def __init__(self, root, mode, selected_attrs=[], crop_size=None, image_size=128, batch_size=16, data_augmentation=False, mask_input_bg=True):
        if mode not in ['train', 'test']:
            return

        self.root = root
        self.data_augmentation = data_augmentation
        self.image_size = image_size
        self.crop_size = crop_size


        train_trf, val_trf = self.setup_transforms(use_illum)
        if mode == 'train':
            val_set = MaterialDataset(root, 'val', selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg)
            self.val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = MaterialDataset(root, 'train', selected_attrs, transform=train_trf, mask_input_bg=mask_input_bg)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            batch_size=53 #size of the test dataset
            test_set = MaterialDataset(root, 'test', selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
    def setup_transforms(self,use_illum):
        #NOTE alwasys first resize to original size in case we are dealing with images of different sizes
        original_size=self.image_size*2 #TODO should be a parameter

        val_trf = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5,0), std=(0.5, 0.5, 0.5,1))
        ])

        if self.data_augmentation:
            train_trf = T.Compose([
                T.Resize(original_size), #suppose the dataset is of size 256
                #T.RandomHorizontalFlip(0.5), #TODO recode for normals
                #T.RandomVerticalFlip(0.5), #TODO recode for normals
                T.RandomCrop(size=self.crop_size),
                T.RandomResize(low=original_size, high=int(original_size*1.1718)),
                #T.RandomRotation(degrees=(-5, 5)), #TODO recode for normals
                val_trf,
            ])
        else:
            train_trf = T.Compose([
                T.Resize(original_size),
                val_trf,
            ])
        val_trf = T.Compose([
            T.Resize(original_size),
            val_trf,
        ])
        return train_trf, val_trf


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    val_trf = T.Compose([
        T.CenterCrop(240),
        T.Resize(128),
        T.ToTensor(),
        #T.Normalize(mean=(0.5, 0.5, 0.5,0), std=(0.5, 0.5, 0.5,1))
    ])
    trf = T.Compose([
        T.Resize(256), #suppose the dataset is of size 256
        #T.RandomHorizontalFlip(0.5), #TODO recode for normals
        #T.RandomVerticalFlip(0.5), #TODO recode for normals
        T.RandomCrop(size=240),
        T.RandomResize(low=256, high=300),
        #T.RandomRotation(degrees=(-5, 5)), #TODO recode for normals
        val_trf,
    ])

    data_root = '/Users/delanoy/Documents/postdoc/project1_material_networks/dataset/renders_by_geom_ldr/network_dataset/'
    data = MaterialDataset(root=data_root,
                           mode='test',
                           transform=trf,
                           mask_input_bg=True)
    loader = DataLoader(data,  batch_size=1, shuffle=True)
    iter(loader)
    for imgs, normal in loader:
        # from matplotlib import pyplot as plt

        # # for i in range(len(imgs)):
        # #     print (infos[i])
        for i in range(len(imgs)):
            #print(illum[i][:,0,0])
            plt.subplot(1,2,1)
            plt.imshow(imgs[i].permute(1, 2, 0).detach().cpu(),cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(normal[i].permute(1, 2, 0).detach().cpu(),cmap='gray')
            plt.show()
        print("done")
        #input('press key to continue plotting')