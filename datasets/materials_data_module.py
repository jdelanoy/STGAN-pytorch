import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.material import MaterialDataset, DisentangledSampler 


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, img):
        size = np.random.randint(self.low, self.high)
        return transforms.functional.resize(img, size, self.interpolation)


class MaterialDataModule(pl.LightningDataModule):
    def __init__(self, root, attrs, crop_size, image_size, batch_size, num_workers,
                 use_disentangled_sampler=False):
        super(MaterialDataModule, self).__init__()
        self.root = root
        self.attrs = attrs
        self.image_size = image_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_disentangled_sampler=use_disentangled_sampler

    def setup(self, stage):
        train_trf, val_trf = self.setup_transforms()
        self.data_train = MaterialDataset(self.root, 'train', self.attrs,self.use_disentangled_sampler, train_trf)
        self.data_val = MaterialDataset(self.root, 'val', self.attrs,self.use_disentangled_sampler, val_trf)
        self.data_test = MaterialDataset(self.root, 'test', self.attrs,False, val_trf) #never use disentangle sampler for test

    def setup_transforms(self):
        val_trf = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_trf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size=self.crop_size),
            RandomResize(low=256, high=300),
            transforms.RandomRotation(degrees=(-5, 5)),
            val_trf,
        ])

        return train_trf, val_trf

    @property
    def sampler(self):
        return DisentangledSampler(self.data_train, batch_size=self.batch_size)

    def train_dataloader(self):
        if self.use_disentangled_sampler:
            return DataLoader(self.data_train,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=self.num_workers,
                            sampler=self.sampler)
        else:
            return DataLoader(self.data_train,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=self.num_workers,
                            drop_last=True,
                            shuffle=True)

    def val_dataloader(self):
        if self.use_disentangled_sampler:
            return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          sampler=self.sampler)
        else:
            return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          shuffle=False)

