import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.unet import Autoencoder, Encoder


class IGNWithGAN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()

        # networks
        self.G = Autoencoder(
            in_ch=3,
            out_ch=3,
            ch=hparams.ch_G,
            norm=hparams.norm_G,
            act=hparams.act_G)
        self.D = nn.Sequential(
            Encoder(
                in_ch=3,
                ch=hparams.ch_D,
                act=hparams.act_D,
                norm=hparams.norm_D,
                return_last=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # store here generated images


    def forward(self, img, mode):
        img_hat = self.G(img, mode)
        return img_hat

    def generator_step(self, batch):
        img, label, mode = batch
        loss = {}

        # get bottleneck
        bneck, enc_feat = self.G.forward_encoder(img)
        bneck_mater, bneck_shape, bneck_illum = self.split_bneck(bneck)

        # Compute invariancy losses
        loss_shape = 0
        loss_illum = 0
        loss_mater = 0

        # only MATERIAL changes in the batch
        if torch.all(mode == 0):
            loss_shape = torch.pdist(bneck_shape, p=2).mean()
            loss_illum = torch.pdist(bneck_illum, p=2).mean()
        # only GEOMETRY changes in the batch
        elif torch.all(mode == 1):
            loss_illum = torch.pdist(bneck_illum, p=2).mean()
            loss_mater = torch.pdist(bneck_mater, p=2).mean()
        # only ILLUMINATION changes in the batch
        elif torch.all(mode == 2):
            loss_shape = torch.pdist(bneck_shape, p=2).mean()
            loss_mater = torch.pdist(bneck_mater, p=2).mean()
        else:
            raise ValueError('data sampling mode not understood')

        # rebuild bneck and reconstruct img
        bneck = self.join_bneck(bneck_mater, bneck_shape, bneck_illum)
        img_hat = self.model.forward_decoder(img, bneck, enc_feat)

        # compute bneck invariancy loss
        loss_inv = (loss_shape + loss_illum + loss_mater) * self.hparams.lambda_G_features
        loss['G/loss_invariancy'] = loss_inv

        # compute L1 losses
        loss['G/loss_l1'] = F.l1_loss(img_hat, img) * self.hparams.lambda_G_l1

        # compute perceptual losses on the generated image
        if self.hparams.lambda_G_perc > 0 or self.hparams.lambda_G_style > 0:
            f_img = self.vgg16_f(img)
            f_img_hat = self.vgg16_f(img_hat)
            loss['G/loss_perc'] = self.hparams.lambda_G_perc * self.loss_P(f_img_hat, f_img)
            loss['G/loss_style'] = self.hparams.lambda_G_style * self.loss_S(f_img_hat, f_img)

        # GAN loss
        y = torch.ones(img_hat.size(0), 1, device=self.device)
        loss['G/loss_GAN'] = F.binary_cross_entropy(self.D(img_hat), y)

        self.log_dict(loss)
        final_loss = torch.stack([v for v in loss.values()]).sum()

        return final_loss

    def discriminator_step(self, batch):
        img, label, mode = batch
        loss = {}

        # train discriminator on real
        y_real = torch.ones(img.size(0), 1, device=self.device)

        # calculate real score
        D_output = self.D(img)
        loss['D/loss_GAN_real'] = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        D_output = self.D(x_fake)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), self.hparams.lr_G, betas=self.hparams.betas_G)
        opt_d = torch.optim.Adam(self.D.parameters(), self.hparams.lr_D, betas=self.hparams.betas_D)
        return [opt_g, opt_d], []

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
            parser.add_argument('--bneck-layers', default=2, type=int)
            parser.add_argument('--norm-bneck', default=True, type=bool)
            parser.add_argument('--ch_G', default=[64, 128, 256], type=int)
            parser.add_argument('--ch_D', default=[64, 128, 256], type=int)
            parser.add_argument('--norm_G', default='batch', type=str)
            parser.add_argument('--norm_D', default='batch', type=str)
            parser.add_argument('--act_G', default='leaky_relu', type=str)
            parser.add_argument('--act_D', default='leaky_relu', type=str)

            # losses weights parameters
            parser.add_argument('--lambda-G-l1', default=1, type=float)
            parser.add_argument('--lambda-G-perc', default=0.1, type=float)
            parser.add_argument('--lambda-G-style', default=0.1, type=float)
            parser.add_argument('--lambda-G-features', default=0.01, type=float)
            parser.add_argument('--use-IGN-grad', dest='use_IGN_grad', default=False,
                                action='store_true')
            parser.add_argument('--do-mean-features', dest='do_mean_features', default=False,
                                action='store_true')

            # optimizer parameters
            parser.add_argument('--optimizer', default='adam', type=str)
            parser.add_argument('--lr_G', default=0.0002, type=float)
            parser.add_argument('--lr_D', default=0.0002, type=float)
            parser.add_argument('--betas_G', default=(0.5, 0.999), type=float)
            parser.add_argument('--betas_D', default=(0.5, 0.999), type=float)

            # logging parameters
            parser.add_argument('--img-log-iter', default=500, type=str)

            # data params
            parser.add_argument('--data-path', default='../data/renders_materials_manu')
            parser.add_argument('--image-size', default=128)
            parser.add_argument('--crop-size', default=240)
            parser.add_argument('--attrs', default=['glossy'])

            return parser

        return __model_args(parent_parser)
