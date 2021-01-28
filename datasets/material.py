import os
import math
import torch
from torch.utils import data
#from torchvision import transforms as T
import datasets.transforms as T
from PIL import Image
import random
import numpy as np
from utils.im_util import get_alpha_channel
import cv2
from matplotlib import pyplot as plt


def make_dataset(root, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines_train = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_train.txt'), 'r')]
    lines_test = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_test.txt'), 'r')]
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
        np.random.shuffle(lines_test)
        #lines = lines_test+random.sample(lines_train,16) #for spheres
        lines = lines_test[:32*2]+random.sample(lines_train,32*4) #for full dataset

        # #only from one shape/one env
        # shape=""
        # env=""
        # lines_train=[line for line in lines_train if (shape in line and env in line)]
        # #take 100 random images
        # np.random.shuffle(lines)
        # lines_train=lines_train[:200]
    #print(len(lines))

    files = []
    mat_attrs = []
    material = []
    geometry = []
    illumination = []
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
        material.append(filename_split[1])
        geometry.append(filename_split[0])
        illumination.append(filename_split[2])

    return {'files': files,
            'mat_attrs': mat_attrs,
            'materials': material,
            'geometries': geometry,
            'illuminations': illumination}


class DisentangledSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.mats = self.data_source.mats
        self.geoms = self.data_source.geoms
        self.illums = self.data_source.illums

        self.batch_size = batch_size

    def __iter__(self):
        # get all possible idxs in the dataset shuffled
        rand_idxs = torch.randperm(len(self.data_source)).tolist()

        # here we keep track of the number of batches sampled
        while len(rand_idxs) > 0:
            # mode for the batch -> 0:material, 1:geometry, 2:illumination
            mode = random.randint(0, 2)

            # get current index
            curr_idx = rand_idxs.pop(0)

            # make structure where we will append the indexes
            yield curr_idx , mode

            # get current object properties
            material_idx = self.mats[curr_idx]
            geometry_idx = self.geoms[curr_idx]
            illumination_idx = self.illums[curr_idx]

            # see other objects in the dataset that we can sample according to the
            # given mode
            if mode == 0:  # only MATERIAL changes in the batch
                materials = self.mats != material_idx
                geometries = self.geoms == geometry_idx
                illumination = self.illums == illumination_idx
            if mode == 1:  # only GEOMETRY changes in the batch
                materials = self.mats == material_idx
                geometries = self.geoms != geometry_idx
                illumination = self.illums == illumination_idx
            if mode == 2:  # only ILLUMINATION changes in the batch
                materials = self.mats == material_idx
                geometries = self.geoms == geometry_idx
                illumination = self.illums != illumination_idx

            # get the intersection of the possible factors to sample
            possible_idx = materials * geometries * illumination

            # retrieve is position in the tensor
            possible_idx = possible_idx.nonzero()

            # randomly shuffle the idxs that are possible to sample
            possible_idx = possible_idx[torch.randperm(len(possible_idx))]

            # populate the batch with such idxs
            for i in range(self.batch_size - 1):
                yield possible_idx[i] , mode

    def __len__(self):
        return len(self.data_source)


def list2idxs(l):
    idx2obj = list(set(l))
    idx2obj.sort()
    obj2idx = {mat: i for i, mat in enumerate(idx2obj)}
    l = [obj2idx[obj] for obj in l]
    #print(idx2obj)
    #print(obj2idx)
    return l, idx2obj, obj2idx


class MaterialDataset(data.Dataset):
    def __init__(self, root, mode, selected_attrs, disentangled=False, transform=None):
        items = make_dataset(root, mode, selected_attrs)

        self.files = items['files']
        self.mat_attrs = items['mat_attrs']

        mats, self.idx2mats, self.mats2idxs = list2idxs(items['materials'])
        geoms, self.idx2geoms, self.geoms2idx = list2idxs(items['geometries'])
        illums, self.idx2illums, self.illums2idx = list2idxs(items['illuminations'])

        self.mats = np.array(mats)
        self.geoms = np.array(geoms)
        self.illums = np.array(illums)

        self.root = root
        self.mode = mode
        self.disentangled=disentangled
        self.transform = transform

    def __getitem__(self, index_and_mode):
        if self.disentangled:
            index, sampling_mode = index_and_mode
        else:
            index = index_and_mode
        
        mat_attr = self.mat_attrs[index]
        #mat=self.mats[index]
        #geom=self.geoms[index]
        #illum=self.illums[index]

        #image = Image.open(os.path.join(self.root, "renderings", self.files[index]))
        image_rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, "renderings", self.files[index]), 1), cv2.COLOR_BGR2RGB)
        size=image_rgb.shape[0]
        #read normals if it exists (otherwise, put the image), and the mask
        normals_bgra = cv2.imread(os.path.join(self.root, "normals", self.files[index][:-3]+"png"), -1)
        if (type(normals_bgra) is np.ndarray):
            normals = np.ndarray((size,size,4), dtype=np.uint8)
            cv2.mixChannels([normals_bgra], [normals], [0,2, 1,1, 2,0, 3,3])
            #mask=normals[:,:,3:]
            #normals = cv2.cvtColor(normals[:,:,:3], cv2.COLOR_BGR2RGB)
            #normals = np.concatenate((normals,mask),2)
        else:
            mask=np.ones((size,size,1),np.uint8)*255
            normals = np.ndarray((size,size,4), dtype=np.uint8)
            cv2.mixChannels([image_rgb,mask], [normals], [0,0, 1,1, 2,2, 3,3])
            # normals=image
            # normals = np.concatenate((normals,mask),2)
        #image = np.concatenate((image,mask),2)
        image = np.ndarray(normals.shape, dtype=np.uint8)
        cv2.mixChannels([image_rgb,normals], [image], [0,0, 1,1, 2,2, 6,3])
        # try:
        #     normals = Image.open(os.path.join(self.root, "normals", self.files[index][:-3]+"png"))
        #     mask=get_alpha_channel(normals)
        # except FileNotFoundError:
        #     #put the original image in place of the normals + full mask
        #     normals=image
        #     mask = Image.new('L',normals.size,255)
        #     normals.putalpha(mask)
        # #print(self.files[index],"before",np.array(image)[0,0,:])
        # image.putalpha(mask)
        # print(self.files[index],"after",np.array(image)[0,0,:])

        #normals.putalpha(mask)
        if self.transform is not None:
            #concatenate everything
            image,normals = self.transform(image,normals) 
        #print(self.files[index],"after trans",np.array(image)[:,0,0])

        illum=image

        if self.disentangled:
            return image,normals,illum, torch.FloatTensor(mat_attr), sampling_mode
        else:
            return image,normals,illum, torch.FloatTensor(mat_attr)

    def __len__(self):
        return len(self.files)



class MaterialDataLoader(object):
    def __init__(self, root, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16, data_augmentation=False):
        if mode not in ['train', 'test']:
            return

        self.root = root
        self.data_augmentation = data_augmentation
        self.image_size = image_size
        self.crop_size = crop_size


        train_trf, val_trf = self.setup_transforms()
        if mode == 'train':
            print("loading data")
            val_set = MaterialDataset(root, 'val', selected_attrs, transform=val_trf)
            self.val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = MaterialDataset(root, 'train', selected_attrs, transform=train_trf)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_set = MaterialDataset(root, 'test', selected_attrs, transform=val_trf)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
    def setup_transforms(self):
        val_trf = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5,0), std=(0.5, 0.5, 0.5,1))
        ])
        if self.data_augmentation:
            train_trf = T.Compose([
                #T.RandomHorizontalFlip(), #TODO recode for normals
                #T.RandomVerticalFlip(), #TODO recode for normals
                T.RandomCrop(size=self.crop_size),
                T.RandomResize(low=256, high=300),
                #T.RandomRotation(degrees=(-5, 5)), #TODO recode for normals
                val_trf,
            ])
        else:
            train_trf = val_trf

        return train_trf, val_trf


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_root = '/Users/delanoy/Documents/postdoc/project1_material_networks/dataset/renders_by_geom_ldr/'
    data = MaterialDataset(root=data_root,
                           mode='train',
                           selected_attrs=['glossy'],
                           transform=transforms.ToTensor())
    sampler = DisentangledSampler(data, batch_size=8)
    loader = DataLoader(data,  batch_size=8, shuffle=True)
    iter(loader)
    for imgs, labels, infos in loader:
        from matplotlib import pyplot as plt

        for i in range(len(imgs)):
            print (infos[i])
        # for img in imgs:
        #     toplot = img.permute(1, 2, 0).detach().cpu()
        #     plt.imshow(toplot)
        #     plt.show()

        input('press key to continue plotting')
