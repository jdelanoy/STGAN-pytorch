import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.optim import lr_scheduler

from models.unet import Autoencoder
from models.stgan import Generator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor




class OriginalIGN(pl.LightningModule):

    def __init__(self, hparams):
        super(OriginalIGN, self).__init__()

        # read experiment params
        self.hparams = hparams

        # create model
        #self.model = Autoencoder(3, 3, hparams.ch, norm=hparams.norm, act=hparams.act,ign_grad=self.hparams.use_IGN_grad)
        self.model = Generator(1,conv_dim=64,n_layers=5,max_dim=512, shortcut_layers=0,n_attr_deconv=0, vgg_like=False)
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
            torch.rand(4, 3, 256, 256).clamp(-1, 1),
            torch.ones(4, 1),
            torch.ones(4, 1),
        ]
    def test_step(self, batch, batch_nb):
        self.test_disentangle(batch)

    def training_step(self, batch, batch_nb):
        img, label, mode = batch

        # get bottleneck
        enc_feat,bneck = self.model.encode(img)
        bneck_size = bneck.shape

        bneck_mater, \
        bneck_shape, \
        bneck_illum = self.model.split_bneck(bneck)
        bneck = [bneck_mater, bneck_shape, bneck_illum]
        # Compute invariancy losses
        loss_shape = 0
        loss_illum = 0
        loss_material = 0

        # clamp features to be equal to the batch mean and compute loss
        if torch.all(mode == 0):  # only MATERIAL changes in the batch
            bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
            bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
            if self.hparams.do_mean_features : bneck = [bneck_mater, bneck_shape_mean, bneck_illum_mean]
            loss_shape = torch.dist(bneck_shape,bneck_shape_mean, p=2).mean()
            loss_illum = torch.dist(bneck_illum,bneck_illum_mean, p=2).mean()
        elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
            bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
            bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
            if self.hparams.do_mean_features : bneck = [bneck_mater_mean, bneck_shape, bneck_illum_mean]
            loss_illum = torch.dist(bneck_illum,bneck_illum_mean, p=2).mean()
            loss_material = torch.dist(bneck_mater,bneck_mater_mean, p=2).mean()
        elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
            bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
            bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
            if self.hparams.do_mean_features : bneck = [bneck_mater_mean, bneck_shape_mean, bneck_illum]
            loss_shape = torch.dist(bneck_shape,bneck_shape_mean, p=2).mean()
            loss_material = torch.dist(bneck_mater, bneck_mater_mean,p=2).mean()
        else:
            raise ValueError('data sampling  mode not understood')
        features_dist={"G/dist_shape":loss_shape,"G/dist_illum":loss_illum,"G/dist_material":loss_material}
        self.log_dict(features_dist, on_step=True, on_epoch=False)

        # join bneck
        bneck = self.model.join_bneck(bneck)

        # compute L1 losses and perceptual losses on the generated image
        img_hat = self.model.decode(bneck, label, enc_feat)

        loss = {}
        
        if self.hparams.lambda_G_features > 0:
            loss['G/loss_invariancy'] = (loss_shape + loss_illum + loss_material)*self.hparams.lambda_G_features
            
        self.reconstruction_loss(img_hat, img,loss)

        loss['loss'] = torch.stack([v for v in loss.values()]).sum()
        self.log_dict(loss, on_step=True, on_epoch=False)
        return loss




    def validation_step(self, batch, batch_nb):
        img, label, mode = batch
        batch_size = img.size(0)

        # get bottleneck
        enc_feat,bneck = self.model.encode(img)
        bneck_size = bneck.shape

        bneck_mater, \
        bneck_shape, \
        bneck_illum = self.model.split_bneck(bneck)

        all_recon = []
        loss = {}
        for ix in range(batch_size):
            # we do not need to swap properties if we do not log them
            if batch_nb % self.hparams.img_log_iter != 0 and ix > 0:
                break

            # clamp features to be equal to the batch mean
            if torch.all(mode == 0):  # only MATERIAL changes in the batch
                # bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(
                #     bneck_shape)
                # bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(
                #     bneck_illum)
                bneck = [bneck_mater, bneck_shape, bneck_illum]
                bneck_mater = torch.roll(bneck_mater, 1, dims=0)
            elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
                # bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(
                #     bneck_mater)
                # bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(
                #     bneck_illum)
                bneck = [bneck_mater, bneck_shape, bneck_illum]
                bneck_shape = torch.roll(bneck_shape, 1, dims=0)
            elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
                # bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
                # bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
                bneck = [bneck_mater, bneck_shape, bneck_illum]
                bneck_illum = torch.roll(bneck_illum, 1, dims=0)
            else:
                raise ValueError('data sampling  mode not understood')

            # reconstruct image
            bneck = self.model.join_bneck(bneck)
            img_hat = self.model.decode(bneck, label, enc_feat)
            all_recon.append(img_hat)

            # compute the loss only when we reconstruct the input image (not for the
            # images reconstructed when properties have been shifted
            if ix == 0:
                self.reconstruction_loss(img_hat, img,loss)
                loss['loss'] = torch.stack([v for v in loss.values()]).sum()
                
        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_nb % self.hparams.img_log_iter == 0:
            all_recon = torch.cat((img, *all_recon), dim=-2)
            img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
            self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)

        return loss

    def test_disentangle(self, batch):
        # forward through the model and get loss
        img, label, mode = batch
        batch_size = img.size(0)

        enc_feat,bneck = self.model.encode(img)
        #bneck_material, bneck_shape, bneck_illum = self.split_bneck(bneck)
        splitted_bneck = self.model.split_bneck(bneck)
        #for each property
        for label in range(len(splitted_bneck)):
            #add image in the left top corner
            x_fake_list = [torch.cat((img[0].unsqueeze(0),img),dim=0)]
            #each image makes a column: all share the same embedding for the branch
            for c in range(batch_size):
                encodings_copy = [enc.clone() for enc in splitted_bneck] #[[enc.clone() for enc in encs] for encs in encodings]
                common_features=encodings_copy[label][c]
                #change encoding for all images
                for i in range(batch_size): #for each image
                    encodings_copy[label][i]=common_features
                # for layer in range(len(common_features)): #for each layer
                #     for i in range(img.shape[0]): #for each image
                #         encodings_copy[label][layer][i]=common_features[layer][c]
                #decode for the encodings in encodings_copy
                bneck = self.model.join_bneck(encodings_copy)
                fake_image=(self.model.decode(bneck, label, enc_feat))
                #add reference image
                fake_image=torch.cat((img[c].unsqueeze(0),fake_image),dim=0)
                x_fake_list.append(fake_image)
            x_concat = torch.cat(x_fake_list, dim=3)
            img_log = tvutils.make_grid(x_concat * 0.5 + 0.5, nrow=1)
            self.tb_add_image('Reconstructed img_'+str(label), img_log, global_step=self.global_step)


    def log_img_interpolation(self, batch):
        # forward through the model and get loss
        img, label, mode = batch
        batch_size = img.size(0)

        enc_feat, bneck = self.model.encode(img)
        bneck_material, bneck_shape, bneck_illum = self.model.split_bneck(bneck)

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

            bneck = self.model.join_bneck([bneck_material, bneck_shape, bneck_illum])
            all_recon.append(self.model.decode(bneck, label, enc_feat))

        all_recon = torch.cat((img, *all_recon), dim=-2)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
        self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)

    def forward(self, img, label, mode):
        img_hat = self.model(img, mode)
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

    # def split_bneck(self, bneck, do_norm=False):
    #     self.bneck_shape = bneck.shape
    #     batch_size = bneck.size(0)
    #     step = bneck.size(1) // 3

    #     bneck_material = bneck[:, :step].view(batch_size, -1)  # 1/3 of the features
    #     bneck_shape = bneck[:, step:step * 2].view(batch_size, -1)
    #     bneck_illum = bneck[:, step * 2:].view(batch_size, -1)

    #     if do_norm:
    #         bneck_material = bneck_material / bneck_material.norm(p=2)
    #         bneck_shape = bneck_shape / bneck_shape.norm(p=2)
    #         bneck_illum = bneck_illum / bneck_illum.norm(p=2)

    #     return bneck_material, bneck_shape, bneck_illum

    # def join_bneck(self, *bnecks):
    #     return torch.cat(bnecks, dim=1).view(*self.bneck_shape)

    def reconstruction_loss(self,img_hat, img,loss):
        loss['G/loss_l1'] = F.l1_loss(img_hat, img)*self.hparams.lambda_G_l1
        if (self.hparams.lambda_G_perc > 0 or self.hparams.lambda_G_style>0):
            f_img = self.vgg16_f(img)
            f_img_hat = self.vgg16_f(img_hat)
            loss['G/loss_perc'] = self.hparams.lambda_G_perc * self.loss_P(f_img_hat, f_img)
            loss['G/loss_style'] = self.hparams.lambda_G_style * self.loss_S(f_img_hat, f_img)


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
            parser.add_argument('--norm-bneck', default=True, type=bool)

            # losses weights parameters
            parser.add_argument('--lambda_G_l1', default=1, type=float)
            parser.add_argument('--lambda_G_perc', default=0.1, type=float)
            parser.add_argument('--lambda_G_style', default=0.1, type=float)
            parser.add_argument('--lambda_G_features', default=0.01, type=float)
            parser.add_argument('--use_IGN_grad', dest='use_IGN_grad', default=False, action='store_true')
            parser.add_argument('--do_mean_features', dest='do_mean_features', default=False, action='store_true')
            
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