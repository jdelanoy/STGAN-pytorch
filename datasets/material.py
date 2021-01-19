import os
import math
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import random
import numpy as np


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, img):
        size = np.random.randint(self.low, self.high)
        return transforms.functional.resize(img, size, self.interpolation)


def make_dataset(root, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines_train = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_train.txt'), 'r')]
    lines_test = [line.rstrip() for line in open(os.path.join(root,  'attributes_dataset_test.txt'), 'r')]
    all_attr_names = lines_train[0].split()
    print(mode,all_attr_names)
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
        lines = lines_test[:6]+random.sample(lines_train,6) #for full dataset

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
        filename_split = filename.split('@')[1].split('.')[0].split('_')
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

        self.root = os.path.join(root, '256px_dataset')
        self.mode = mode
        self.disentangled=disentangled
        self.transform = transform

    def __getitem__(self, index_and_mode):
        if self.disentangled:
            index, sampling_mode = index_and_mode
        else:
            index = index_and_mode
        
        filename = os.path.join(self.root, self.files[index])
        mat_attr = self.mat_attrs[index]
        #mat=self.mats[index]
        #geom=self.geoms[index]
        #illum=self.illums[index]

        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        #TODO also load normals/illum and apply the exact same transformation
        normals=image
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

        transform = []
        if crop_size is not None:
            transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if mode == 'train':
            print("loading data")
            val_transform = transforms.Compose(transform)       # make val loader before transform is inserted
            val_set = MaterialDataset(root, 'val', selected_attrs, transform=val_transform)
            #sampler = DisentangledSampler(val_set, batch_size=batch_size)
            #self.val_loader = data.DataLoader(val_set, batch_size=batch_size, sampler=sampler, num_workers=4)
            self.val_loader = data.DataLoader(val_set, batch_size=20, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            if data_augmentation:
                transform.insert(0, transforms.RandomHorizontalFlip())
                transform.insert(0, transforms.RandomVerticalFlip())
                transform.insert(0, transforms.RandomCrop(size=crop_size))
                transform.insert(0, RandomResize(low=256, high=300))
                transform.insert(0, transforms.RandomRotation(degrees=(-5, 5)))
            train_transform = transforms.Compose(transform)
            train_set = MaterialDataset(root, 'train', selected_attrs, transform=train_transform)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_transform = transforms.Compose(transform)
            test_set = MaterialDataset(root, 'test', selected_attrs, transform=test_transform)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))

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
