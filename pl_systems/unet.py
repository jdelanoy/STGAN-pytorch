import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.optim import lr_scheduler

from models.networks import Unet
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor


class NormalUnet(pl.LightningModule):

    def __init__(self, hparams):
        super(NormalUnet, self).__init__()

        # read experiment params
        self.hparams = hparams

        # create model
        self.model = Unet(conv_dim=hparams.conv_dim,n_layers=hparams.n_layers,max_dim=hparams.max_dim, skip_connections=hparams.skip_connections, vgg_like=0)
        print(self.model)

        # dicts to store the images during train/val steps
        self.last_val_batch = {}
        self.last_val_pred = {}
        self.last_train_batch = {}
        self.last_train_pred = {}

        # create all the loss functions that we may need
        self.L1 = torch.nn.L1Loss()
        self.loss_P = PerceptualLoss()
        self.loss_S = StyleLoss()
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4'])


        self.example_input_array = [
            torch.rand(4, 3, 128, 128).clamp(-1, 1)
        ]

    def test_step(self, batch, batch_nb):
        print("not implemented yet")

    def shared_step(self, batch):
        img, normals, _, _ = batch
        normals_hat = self.model(img)

        loss = {}
        loss['reconstruction']=F.l1_loss(normals_hat, normals) #TODO change the loss
        return loss, normals_hat

    def training_step(self, batch, batch_nb):
        loss,_=self.shared_step(batch)
        loss['loss'] = torch.stack([v for v in loss.values()]).sum()
        self.log_dict(loss, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_nb):
        loss,normals_hat=self.shared_step(batch)

        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_nb % self.hparams.img_log_iter == 0:
            self.tb_add_image('Reconstructed img', normals_hat * 0.5 + 0.5, global_step=self.global_step)

        return loss

    def forward(self, img, normals, illum, mat_attr):
        img_hat = self.model(img)
        return img_hat

    def on_epoch_end(self):
        self.eval()

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=0.0001, betas=(0.9, 0.999))
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9,
                                        weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')
        scheduler = lr_scheduler.StepLR(optimizer, 50000)
        # scheduler = {
        #     'scheduler': lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer,
        #         patience=5,
        #         factor=0.1),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        return [optimizer], [scheduler]


    @property
    def tb_add_image(self):
        # easier call to log images
        return self.logger.experiment.add_image

    @staticmethod
    def add_ckpt_args(parent_parser):
        def __model_args(parent_parser):
            parser = argparse.ArgumentParser(parents=[parent_parser])

            # run paremeters
            parser.add_argument('--gpus', type=str, default='-1')
            parser.add_argument('--seed', type=int, default=1953)

            # dataset parameters
            parser.add_argument('--num-workers', default=10, type=int)
            parser.add_argument('--batch-size', default=6, type=int)

            # model cfg parameters
            parser.add_argument('--conv_dim', default=64, type=int)
            parser.add_argument('--n_layers', default=6, type=int)
            parser.add_argument('--max_dim', default=512, type=int)
            parser.add_argument('--skip-connections', default=2, type=int)

            # optimizer parameters
            parser.add_argument('--optimizer', default='adam', type=str)

            # logging parameters
            parser.add_argument('--img-log-iter', default=500, type=str)

            # data params
            parser.add_argument('--data-path', default='../data/renders_materials_manu')
            parser.add_argument('--image-size', default=128)
            parser.add_argument('--crop-size', default=240)

            return parser

        return __model_args(parent_parser)
