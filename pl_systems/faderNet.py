import argparse
import numpy as np
import cv2

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
        print(self.latent_disc)
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
        encodings,z = self.model.encode(img)
        img_hat = self.model.decode(mat_attr,z,encodings)
        #go through latent disc
        out_att = self.latent_disc(z)

        #train generator
        if optimizer_idx == 0:
            loss['G/loss_latent'] = -self.regression_loss(out_att, mat_attr) * self.hparams.lambda_G_latent
            self.reconstruction_loss(img, img_hat,loss) 
            loss['G/loss'] = torch.stack([v for v in loss.values()]).sum()
            self.log_dict(loss, on_step=True, on_epoch=False)
            return loss['G/loss']
        #train latent disc
        if optimizer_idx == 1:
            loss['LD/loss_latent'] = self.regression_loss(out_att, mat_attr)*self.hparams.lambda_LD
            loss['LD/loss'] = torch.stack([v for v in loss.values()]).sum()
            self.log_dict(loss, on_step=True, on_epoch=False)
            return loss['LD/loss']


    def validation_step(self, batch, batch_nb):
        img, _, _, mat_attr = batch
        loss = {}
        #go through generator
        encodings,z = self.model.encode(img)
        img_hat = self.model.decode(mat_attr,z,encodings)
        #go through latent disc
        out_att = self.latent_disc(z)

        self.reconstruction_loss(img, img_hat,loss) 
        loss['G/loss'] = torch.stack([v for v in loss.values()]).sum()
        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_nb % self.hparams.img_log_iter == 0:
            #TODO do the attribute interpolation
            #self.log_img_reconstruction(batch)
            self.log_img_attr_interpolation(batch)

        return loss

    #### do not need to be in FaderNet module
    def create_interpolated_attr(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""
        c_trg_list = [c_org]
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg)
        return c_trg_list
    #### do not need to be in FaderNet module
    def write_labels_on_images(self,images, labels):
        for im in range(images.shape[0]):
            text_image=np.zeros((128,128,3), np.uint8)
            for i in range(labels.shape[1]):
                cv2.putText(text_image, "%.2f"%(labels[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
            image_numpy=((text_image.astype(np.float32))/255).transpose(2,0,1)+images[im].cpu().detach().numpy()
            images[im]= torch.from_numpy(image_numpy)

    def log_img_attr_interpolation(self, batch):
        img, _, _, mat_attr = batch

        #interpolated attributes
        all_attr = self.create_interpolated_attr(mat_attr,self.hparams.attrs)
        all_recon = []
        for fake_attr in all_attr:
            img_hat = self.model(img,fake_attr)
            self.write_labels_on_images(img_hat, fake_attr)
            all_recon.append(img_hat)

        all_recon = torch.cat((img, *all_recon), dim=-1)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=1)
        self.tb_add_image('Interpolation img', img_log, global_step=self.global_step)

    def log_img_reconstruction(self, batch):
        # forward through the model and get loss
        img, _, _, mat_attr = batch
        img_hat = self.model(img,mat_attr)

        images=(img,img,img_hat)

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
            optimizer1 = torch.optim.Adam(self.model.parameters(),lr=0.0002, betas=(0.9, 0.999))
            optimizer2 = torch.optim.Adam(self.latent_disc.parameters(),lr=0.00002, betas=(0.9, 0.999))
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer1 = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
            optimizer2 = torch.optim.SGD(self.latent_disc.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')
        scheduler1 = lr_scheduler.StepLR(optimizer1, 100000)
        scheduler2 = lr_scheduler.StepLR(optimizer2, 100000)
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



    def reconstruction_loss(self,img_hat, img,loss):
        loss['G/loss_l1'] = F.mse_loss(img_hat, img)*self.hparams.lambda_G_l1
        if (self.hparams.lambda_G_perc > 0 or self.hparams.lambda_G_style>0):
            f_img = self.vgg16_f(img)
            f_img_hat = self.vgg16_f(img_hat)
            if self.hparams.lambda_G_perc > 0:
                loss['G/loss_perc'] = self.hparams.lambda_G_perc * self.loss_P(f_img_hat, f_img)
            if self.hparams.lambda_G_style > 0:
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
            parser.add_argument('--batch-size', default=32, type=int)

            # model cfg parameters
            parser.add_argument('--conv_dim', default=32, type=int)
            parser.add_argument('--n_layers', default=6, type=int)
            parser.add_argument('--max_dim', default=512, type=int)
            parser.add_argument('--skip-connections', default=0, type=int)

            # optimizer parameters
            parser.add_argument('--optimizer', default='adam', type=str)
            parser.add_argument('--lambda_G_l1', default=1, type=float)
            parser.add_argument('--lambda_G_perc', default=0.1, type=float)
            parser.add_argument('--lambda_G_style', default=0.1, type=float)
            parser.add_argument('--lambda_G_latent', default=1, type=float)
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
