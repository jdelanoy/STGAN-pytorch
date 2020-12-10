import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.optim import lr_scheduler

from models.unet import Autoencoder
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor


class HardIGN(pl.LightningModule):

    def __init__(self, hparams):
        super(HardIGN, self).__init__()

        # read experiment params
        self.hparams = hparams

        # create model
        # self.model = UNet(3, 3, hparams.ch, norm=hparams.norm, act=hparams.act)
        self.model = Autoencoder(3, 3, hparams.ch, norm=hparams.norm, act=hparams.act)

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

        self.example_input_array = torch.rand(4, 3, 256, 256).clamp(-1, 1)

    def training_step(self, batch, batch_nb):
        loss = self.shared_step(batch)
        self.log_dict(loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.shared_step(batch)
        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False)

        if batch_nb % self.hparams.img_log_iter == 0:
            self.log_img_interpolation(batch)

        return loss

    def shared_step(self, batch):
        img, label, mode = batch

        # get bottleneck
        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)

        loss = {}

        # Compute invariancy losses
        loss_shape = 0
        loss_illum = 0
        loss_material = 0

        # only MATERIAL changes in the batch
        if torch.all(mode == 0):
            loss_shape = torch.pdist(bneck_shape, p=2).mean()
            loss_illum = torch.pdist(bneck_illum, p=2).mean()
        # only GEOMETRY changes in the batch
        elif torch.all(mode == 1):
            loss_illum = torch.pdist(bneck_illum, p=2).mean()
            loss_material = torch.pdist(bneck_material, p=2).mean()
        # only ILLUMINATION changes in the batch
        elif torch.all(mode == 2):
            loss_shape = torch.pdist(bneck_shape, p=2).mean()
            loss_material = torch.pdist(bneck_material, p=2).mean()
        else:
            raise ValueError('data sampling mode not understood')

        # compute bneck invariancy loss
        loss['invariancy'] = 0.1 * (loss_shape + loss_illum + loss_material)

        # compute L1 losses and perceptual losses on the generated image
        bneck = self.join_bneck(bneck_material, bneck_shape, bneck_illum)
        img_hat = self.model.forward_decoder(img, bneck, enc_feat)

        loss['l1'] = F.l1_loss(img_hat, img)

        # f_img = self.vgg16_f(img)
        # f_img_hat = self.vgg16_f(img_hat)
        # loss['perc'] = 0.1 * self.loss_P(f_img_hat, f_img)
        # loss['style'] = 0.1 * self.loss_S(f_img_hat, f_img)
        loss['loss'] = torch.stack([v for v in loss.values()]).sum()

        return loss

    def log_img_interpolation(self, batch):
        # forward through the model and get loss
        img, label, mode = batch
        batch_size = img.size(0)

        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)

        all_recon = []
        for shift in range(batch_size):
            # only MATERIAL changes in the batch
            if torch.all(mode == 0):
                bneck_material = torch.roll(bneck_material, shift, dims=0)
            # only GEOMETRY changes in the batch
            elif torch.all(mode == 1):
                bneck_shape = torch.roll(bneck_shape, shift, dims=0)
            # only ILLUMINATION changes in the batch
            elif torch.all(mode == 2):
                bneck_illum = torch.roll(bneck_illum, shift, dims=0)

            bneck = self.join_bneck(bneck_material, bneck_shape, bneck_illum)
            all_recon.append(self.model.forward_decoder(img, bneck, enc_feat))

        all_recon = torch.cat((img, *all_recon), dim=-2)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
        self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)

    def forward(self, img):
        img_hat = self.model(img)
        return img_hat

    def on_epoch_end(self):
        self.eval()

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=0.0002, betas=(0.9, 0.999))
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9,
                                        weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')

        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=5,
                factor=0.1),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def split_bneck(self, bneck, do_norm=False):
        self.bneck_shape = bneck.shape
        batch_size = bneck.size(0)
        step = bneck.size(1) // 3

        bneck_material = bneck[:, :step].view(batch_size, -1)  # 1/3 of the features
        bneck_shape = bneck[:, step:step * 2].view(batch_size, -1)
        bneck_illum = bneck[:, step * 2:].view(batch_size, -1)

        if do_norm:
            bneck_material = bneck_material / bneck_material.norm(p=2)
            bneck_shape = bneck_shape / bneck_shape.norm(p=2)
            bneck_illum = bneck_illum / bneck_illum.norm(p=2)

        return bneck_material, bneck_shape, bneck_illum

    def join_bneck(self, *bnecks):
        return torch.cat(bnecks, dim=1).view(*self.bneck_shape)

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
            parser.add_argument('--ch', default=[64, 128, 256, 512, 512], type=int)
            parser.add_argument('--norm', default='batch', type=str)
            parser.add_argument('--act', default='leaky_relu', type=str)

            # optimizer parameters
            parser.add_argument('--optimizer', default='adam', type=str)

            # logging parameters
            parser.add_argument('--img-log-iter', default=500, type=str)

            # data params
            parser.add_argument('--data-path', default='../data/renders_materials_manu')
            parser.add_argument('--image-size', default=128)
            parser.add_argument('--crop-size', default=240)
            parser.add_argument('--attrs', default=['glossy'])

            return parser

        return __model_args(parent_parser)


class SoftIGN(HardIGN):

    def shared_step(self, batch):
        img, label, mode = batch

        # get bottleneck
        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)

        loss = {}

        # Compute invariancy losses
        loss_shape = 0
        loss_illum = 0
        loss_material = 0

        # only MATERIAL is equal in the batch
        # shift = random.randint(0, batch_size)
        if torch.all(mode == 0):
            loss_material = torch.pdist(bneck_material, p=2).mean()
        # only GEOMETRY changes in the batch
        elif torch.all(mode == 1):
            loss_shape = torch.pdist(bneck_shape, p=2).mean()
        # only ILLUMINATION changes in the batch
        elif torch.all(mode == 2):
            loss_illum = torch.pdist(bneck_illum, p=2).mean()
        else:
            raise ValueError('data sampling mode not understood')

        # compute bneck invariancy loss
        loss['invariancy'] = 0.01 * (loss_shape + loss_illum + loss_material)

        # compute L1 losses and perceptual losses on the generated image
        bneck = self.join_bneck(bneck_material, bneck_shape, bneck_illum)
        img_hat = self.model.forward_decoder(img, bneck, enc_feat)

        loss['l1'] = F.l1_loss(img_hat, img)

        # f_img = self.vgg16_f(img)
        # f_img_hat = self.vgg16_f(img_hat)
        # loss['perc'] = 0.1 * self.loss_P(f_img_hat, f_img)
        # loss['style'] = 0.1 * self.loss_S(f_img_hat, f_img)
        loss['loss'] = torch.stack([v for v in loss.values()]).sum()

        return loss

    def log_img_interpolation(self, batch):
        # forward through the model and get loss
        img, label, mode = batch
        batch_size = img.size(0)

        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)

        all_recon = []
        for shift in range(batch_size):
            # only MATERIAL changes in the batch
            if torch.all(mode == 0):
                bneck_material = torch.roll(bneck_material, shift, dims=0)
            # only GEOMETRY changes in the batch
            elif torch.all(mode == 1):
                bneck_shape = torch.roll(bneck_shape, shift, dims=0)
            # only ILLUMINATION changes in the batch
            elif torch.all(mode == 2):
                bneck_illum = torch.roll(bneck_illum, shift, dims=0)

            bneck = self.join_bneck(bneck_material, bneck_shape, bneck_illum)
            all_recon.append(self.model.forward_decoder(img, bneck, enc_feat))

        all_recon = torch.cat((img, *all_recon), dim=-2)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
        self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)

    def forward(self, img):
        img_hat = self.model(img)
        return img_hat


class OriginalIGN(HardIGN):

    def training_step(self, batch, batch_nb):
        img, label, mode = batch
        optimizer = self.optimizers()

        # get bottleneck
        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_mater, bneck_shape, bneck_illum = self.split_bneck(bneck)

        # clamp features to be equal to the batch mean
        if torch.all(mode == 0):  # only MATERIAL changes in the batch
            bneck_shape = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
            bneck_illum = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
        elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
            bneck_mater = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
            bneck_illum = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
        elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
            bneck_shape = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
            bneck_mater = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
        else:
            raise ValueError('data sampling  mode not understood')

        # compute L1 losses and perceptual losses on the generated image
        bneck = self.join_bneck(bneck_mater, bneck_shape, bneck_illum)
        bneck.retain_grad()
        img_hat = self.model.forward_decoder(img, bneck, enc_feat)

        # f_img = self.vgg16_f(img)
        # f_img_hat = self.vgg16_f(img_hat)

        loss = {}
        loss['l1'] = F.l1_loss(img_hat, img)
        # loss['perc'] = 0.1 * self.loss_P(f_img_hat, f_img)
        # loss['style'] = 0.1 * self.loss_S(f_img_hat, f_img)
        loss['loss'] = torch.stack([v for v in loss.values()]).sum()

        # manually do the backward in the loss
        self.manual_backward(loss['loss'], optimizer, retain_graph=True)

        # retreive bneck gradients
        bneck_mater_dx, bneck_shape_dx, bneck_illum_dx = self.split_bneck(bneck.grad)

        # clamp gradients to be equal to the mean difference
        # from S3.2 IGN paper: "we train all the neurons which correspond to the inactive
        # transformations with an error gradient equal to their difference from the mean.
        # [...] This regularizing force needs to be scaled to be much smaller than the
        # true training signal, otherwise it can overwhelm the reconstruction goal.
        # Empirically, a factor of 1/100 works well."
        if torch.all(mode == 0):  # only MATERIAL changes in the batch
            bneck_shape_dx = 1/100 * (bneck_shape_dx - bneck_shape)
            bneck_illum_dx = 1/100 * (bneck_illum_dx - bneck_illum)
        elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
            bneck_mater_dx = 1/100 * (bneck_mater_dx - bneck_mater)
            bneck_illum_dx = 1/100 * (bneck_illum_dx - bneck_illum)
        elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
            bneck_shape_dx = 1/100 * (bneck_shape_dx - bneck_shape)
            bneck_mater_dx = 1/100 * (bneck_mater_dx - bneck_mater)
        else:
            raise ValueError('data sampling mode not understood')

        # join gradients back together
        bneck.grad = self.join_bneck(bneck_mater_dx, bneck_shape_dx, bneck_illum_dx)

        optimizer.step()

    def validation_step(self, batch, batch_nb):
        pass

    def log_img_interpolation(self, batch):
        # forward through the model and get loss
        img, label, mode = batch
        batch_size = img.size(0)

        bneck, enc_feat = self.model.forward_encoder(img)
        bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)

        all_recon = []
        for shift in range(batch_size):
            # only MATERIAL changes in the batch
            if torch.all(mode == 0):
                bneck_material = torch.roll(bneck_material, shift, dims=0)
            # only GEOMETRY changes in the batch
            elif torch.all(mode == 1):
                bneck_shape = torch.roll(bneck_shape, shift, dims=0)
            # only ILLUMINATION changes in the batch
            elif torch.all(mode == 2):
                bneck_illum = torch.roll(bneck_illum, shift, dims=0)

            bneck = self.join_bneck(bneck_material, bneck_shape, bneck_illum)
            all_recon.append(self.model.forward_decoder(img, bneck, enc_feat))

        all_recon = torch.cat((img, *all_recon), dim=-2)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
        self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)

    def forward(self, img):
        img_hat = self.model(img)
        return img_hat
