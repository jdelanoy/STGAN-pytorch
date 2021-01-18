import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.optim import lr_scheduler

from models.networks import FaderNetGenerator, Latent_Discriminator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor


class FaderNet(pl.LightningModule):

    def __init__(self, hparams):
        super(FaderNet, self).__init__()

        # read experiment params
        self.hparams = hparams

        # create model
        self.model = FaderNetGenerator(conv_dim=hparams.conv_dim,n_layers=hparams.n_layers,max_dim=hparams.max_dim, skip_connections=hparams.skip_connections, vgg_like=0, attr_dim=len(hparams.attrs), n_attr_deconv=1)
        self.latent_disc=Latent_Discriminator(image_size=hparams.image_size, attr_dim=len(hparams.attrs), conv_dim=hparams.conv_dim,n_layers=hparams.n_layers,max_dim=hparams.max_dim, skip_connections=hparams.skip_connections,fc_dim=1024, vgg_like=0)
        print(self.model)

        # dicts to store the images during train/val steps
        self.last_val_batch = {}
        self.last_val_pred = {}
        self.last_train_batch = {}
        self.last_train_pred = {}

        # create all the loss functions that we may need
        self.loss_P = PerceptualLoss()
        self.loss_S = StyleLoss()
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4'])


        self.example_input_array = [
            torch.rand(4, 3, 128, 128).clamp(-1, 1),
            torch.rand(4, 3, 128, 128).clamp(-1, 1),
            torch.rand(4, 3, 128, 128).clamp(-1, 1),
            torch.rand(4, 1).clamp(-1, 1)
        ]

    def test_step(self, batch, batch_nb):
        print("not implemented yet")


    def training_step(self, batch, batch_nb, optimizer_idx):
        img, _, _, mat_attr = batch
        loss = {}
        #go through generator
        encodings,z = self.encode(img)
        img_hat = self.decode(mat_attr,z,encodings)
        #go through latent disc
        out_att = self.latent_disc(z)

        #train generator
        if optimizer_idx == 0:
            loss['G/loss_latent'] = -self.regression_loss(out_att, mat_attr) * self.config.lambda_G_latent
            loss['G/loss_rec']=self.reconstruction_loss(img, img_hat) 
            loss['loss'] = torch.stack([v for v in loss.values()]).sum()
            self.log_dict(loss, on_step=True, on_epoch=False)
            return loss
        #train latent disc
        if optimizer_idx == 1:
            loss['LD/loss'] = self.regression_loss(out_att, mat_attr)*self.config.lambda_LD
            loss['loss'] = torch.stack([v for v in loss.values()]).sum()
            self.log_dict(loss, on_step=True, on_epoch=False)
            return loss


    def validation_step(self, batch, batch_nb):
        img, _, _, mat_attr = batch
        loss = {}
        #go through generator
        encodings,z = self.encode(img)
        img_hat = self.decode(mat_attr,z,encodings)
        #go through latent disc
        out_att = self.latent_disc(z)

        loss['G/loss_rec']=self.reconstruction_loss(img, img_hat) 
        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_nb % self.hparams.img_log_iter == 0:
            #TODO do the attribute interpolation
            self.log_img_interpolation(batch)

        return loss

    def log_img_reconstruction(self, batch):
        # forward through the model and get loss
        img, normals, _, _ = batch
        img_hat = self.model(img)

        images=(img,normals,img_hat)

        images = torch.cat(images, dim=-1)

        img_log = tvutils.make_grid(images * 0.5 + 0.5, nrow=1)
        self.tb_add_image('Reconstructed img', img_log, global_step=self.global_step)


    def forward(self, img, normals, illum, mat_attr):
        img_hat = self.model(img, mat_attr)
        return img_hat

    def on_epoch_end(self):
        self.eval()

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'adam':
            optimizer1 = torch.optim.Adam(self.model.parameters(),lr=0.0001, betas=(0.9, 0.999))
            optimizer2 = torch.optim.Adam(self.latent_disc.parameters(),lr=0.0001, betas=(0.9, 0.999))
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer1 = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
            optimizer2 = torch.optim.SGD(self.latent_disc.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')
        scheduler1 = lr_scheduler.StepLR(optimizer1, 50000)
        scheduler2 = lr_scheduler.StepLR(optimizer2, 50000)
        # scheduler = {
        #     'scheduler': lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer,
        #         patience=5,
        #         factor=0.1),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        return [optimizer1, optimizer2], [scheduler1,scheduler2]


    # # Alternating schedule for optimizer steps (ie: GANs)
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     # update generator opt every 2 steps
    #     if optimizer_idx == 0:
    #         if batch_nb % 2 == 0 :
    #         optimizer.step(closure=closure)

    #     # update discriminator opt every 2 steps
    #     if optimizer_idx == 1:
    #         if batch_nb % 2 == 1 :
    #         optimizer.step(closure=closure)  


    def reconstruction_loss(self,img_hat, img,loss):
        loss['G/loss_l1'] = F.l1_loss(img_hat, img)*self.hparams.lambda_G_l1
        if (self.hparams.lambda_G_perc > 0 or self.hparams.lambda_G_style>0):
            f_img = self.vgg16_f(img)
            f_img_hat = self.vgg16_f(img_hat)
            loss['G/loss_perc'] = self.hparams.lambda_G_perc * self.loss_P(f_img_hat, f_img)
            loss['G/loss_style'] = self.hparams.lambda_G_style * self.loss_S(f_img_hat, f_img)
    def regression_loss(self, logit, target):
        return F.l1_loss(logit,target)/ logit.size(0)

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
            parser.add_argument('--lambda_G_l1', default=1, type=float)
            parser.add_argument('--lambda_G_perc', default=0.1, type=float)
            parser.add_argument('--lambda_G_style', default=0.1, type=float)
            parser.add_argument('--lambda_G_features', default=0.01, type=float)
            parser.add_argument('--lambda_G_latent', default=0.01, type=float)
            parser.add_argument('--lambda_LD', default=0.01, type=float)
            # logging parameters
            parser.add_argument('--img-log-iter', default=500, type=str)

            # data params
            parser.add_argument('--data-path', default='../data/renders_materials_manu')
            parser.add_argument('--image-size', default=128)
            parser.add_argument('--crop-size', default=240)
            parser.add_argument('--attrs', default=['glossy'])

            return parser

        return __model_args(parent_parser)
