import argparse
import numpy as np
import cv2

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.optim import lr_scheduler

from models.networks import FaderNetGenerator, Latent_Discriminator, Discriminator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor
from utils.im_util import denorm, write_labels_on_images


class FaderNet(pl.LightningModule):

    def __init__(self, hparams):
        super(FaderNet, self).__init__()

        # read experiment params
        self.hparams = hparams
        print(hparams)

        # create model
        self.G = FaderNetGenerator(conv_dim=hparams.g_conv_dim,n_layers=hparams.g_layers,max_dim=hparams.max_conv_dim, skip_connections=hparams.skip_connections, vgg_like=hparams.vgg_like, attr_dim=len(hparams.attrs), n_attr_deconv=hparams.n_attr_deconv)
        self.D = Discriminator(image_size=hparams.image_size, attr_dim=len(hparams.attrs), conv_dim=hparams.d_conv_dim,n_layers=hparams.d_layers,max_dim=hparams.max_conv_dim,fc_dim=hparams.d_fc_dim)
        self.LD = Latent_Discriminator(image_size=hparams.image_size, attr_dim=len(hparams.attrs), conv_dim=hparams.g_conv_dim,n_layers=hparams.g_layers,max_dim=hparams.max_conv_dim, fc_dim=hparams.d_fc_dim, skip_connections=hparams.skip_connections, vgg_like=hparams.vgg_like)

        print(self.G)
        if self.hparams.use_image_disc:
            print(self.D)
        if self.hparams.use_latent_disc:
            print(self.LD)
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
        # ================================================================================= #
        #                            1. Preprocess input data                               #
        # ================================================================================= #
        Ia, _, _, a_att = batch
        # generate target domain labels randomly
        b_att =  torch.rand_like(a_att)*2-1.0 # a_att + torch.randn_like(a_att)*self.hparams.gaussian_stddev

        scalars = {}
        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.hparams.use_image_disc and optimizer_idx == 1:
            for _ in range(self.hparams.n_critic):
                # input is the real image Ia
                out_disc_real = self.D(Ia)
                # fake image Ib_hat
                Ib_hat = self.G(Ia, b_att)
                out_disc_fake = self.D(Ib_hat.detach())
                #adversarial losses
                d_loss_adv_real = - torch.mean(out_disc_real)
                d_loss_adv_fake = torch.mean(out_disc_fake)
                # compute loss for gradient penalty
                alpha = torch.rand(Ia.size(0), 1, 1, 1)
                x_hat = (alpha * Ia.data + (1 - alpha) * Ib_hat.data).requires_grad_(True)
                out_disc = self.D(x_hat)
                d_loss_adv_gp = self.hparams.lambda_gp * self.gradient_penalty(out_disc, x_hat)
                #full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss = self.hparams.lambda_adv * d_loss_adv
                scalars['D/loss_adv'] = d_loss.item()
                scalars['D/loss_real'] = d_loss_adv_real.item()
                scalars['D/loss_fake'] = d_loss_adv_fake.item()
                scalars['D/loss_gp'] = d_loss_adv_gp.item()

                # summarize
                scalars['D/loss'] = d_loss.item()
            self.log_dict(scalars, on_step=True, on_epoch=False)
            return d_loss

        # ================================================================================= #
        #                        3. Train the latent discriminator (FaderNet)               #
        # ================================================================================= #
        if self.hparams.use_latent_disc and optimizer_idx == 2:
            # compute disc loss on encoded image
            _,bneck = self.G.encode(Ia)

            for _ in range(self.hparams.n_critic_ld):
                out_att = self.LD(bneck)
                #classification loss
                ld_loss = self.regression_loss(out_att, a_att)*self.hparams.lambda_LD
                # summarize
                scalars['LD/loss'] = ld_loss.item()
            self.log_dict(scalars, on_step=True, on_epoch=False)
            return ld_loss

        # ================================================================================= #
        #                              3. Train the generator                               #
        # ================================================================================= #
        if optimizer_idx == 0:
            encodings,bneck = self.G.encode(Ia)

            Ia_hat=self.G.decode(a_att,bneck,encodings)
            g_loss_rec = self.hparams.lambda_G_rec * self.image_reconstruction_loss(Ia,Ia_hat,scalars)
            g_loss = g_loss_rec
            scalars['G/loss_rec'] = g_loss_rec.item()

            #latent discriminator for attribute in the material part TODO mat part only
            if self.hparams.use_latent_disc:
                out_att = self.LD(bneck)
                g_loss_latent = -self.hparams.lambda_G_latent * self.regression_loss(out_att, a_att)
                g_loss += g_loss_latent
                scalars['G/loss_latent'] = g_loss_latent.item()

            if self.hparams.use_image_disc:
                # original-to-target domain : Ib_hat -> GAN + classif
                Ib_hat = self.G(Ia, b_att)
                out_disc = self.D(Ib_hat)
                # GAN loss
                g_loss_adv = - self.hparams.lambda_adv * torch.mean(out_disc)
                g_loss += g_loss_adv
                scalars['G/loss_adv'] = g_loss_adv.item()
            # summarize
            scalars['G/loss'] = g_loss.item()

            self.log_dict(scalars, on_step=True, on_epoch=False)
            return g_loss

    def validation_step(self, batch, batch_nb):
        Ia_sample, _, _, a_sample = batch
        if batch_nb % self.hparams.img_log_iter == 0:
            b_samples = self.create_interpolated_attr(a_sample, self.hparams.attrs)
            self.compute_sample_grid(Ia_sample,b_samples,a_sample,os.path.join(self.hparams.sample_dir, 'sample_{}.jpg'.format(self.global_step)),writer=True)

        loss = {}
        #go through generator
        encodings,z = self.model.encode(img)
        img_hat = self.model.decode(mat_attr,z,encodings)
        self.reconstruction_loss(img, img_hat,loss) 
        loss['G/loss'] = torch.stack([v for v in loss.values()]).sum()
        loss = {'val_%s' % k: v for k, v in loss.items()}
        self.log_dict(loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    ################################################################
    ##################### EVAL UTILITIES ###########################


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

    def compute_sample_grid(self,x_sample,c_sample_list,c_org_sample,path=None,writer=False):
        x_sample = x_sample
        x_fake_list = [x_sample]
        for c_trg_sample in c_sample_list:
            fake_image=self.G(x_sample, c_trg_sample)
            write_labels_on_images(fake_image,c_trg_sample)
            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(x_concat * 0.5 + 0.5, nrow=1)
        if writer:
            self.tb_add_image('sample', image, global_step=self.global_step)
        if path:
            tvutils.save_image(image,path,nrow=1, padding=0)



    def forward(self, img, normals, illum, mat_attr):
        img_hat = self.G(img, mat_attr)
        return img_hat

    def on_epoch_end(self):
        self.eval()

    ################################################################
    ################### OPTIM UTILITIES ############################
    def build_optimizer(self,model,lr):
        if self.hparams.optimizer.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr, [self.hparams.beta1, self.hparams.beta2])
        elif self.hparams.optimizer.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')
    def build_scheduler(self,optimizer,not_load=False):
        return optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_decay_iters, gamma=0.1)
    def configure_optimizers(self):
        self.optimizer_G = self.build_optimizer(self.G, self.hparams.g_lr)
        self.optimizer_D = self.build_optimizer(self.D, self.hparams.d_lr)
        self.optimizer_LD = self.build_optimizer(self.LD, self.hparams.ld_lr)
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.hparams.use_image_disc))
        self.lr_scheduler_LD = self.build_scheduler(self.optimizer_LD, not self.hparams.use_latent_disc)
        # scheduler = {
        #     'scheduler': lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer,
        #         patience=5,
        #         factor=0.1),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        return [self.optimizer_G, self.optimizer_D,self.optimizer_LD], [self.lr_scheduler_G,self.lr_scheduler_D,self.lr_scheduler_LD]




    ################################################################
    ################### LOSSES UTILITIES ###########################
    def regression_loss(self, logit, target): #static
        return F.l1_loss(logit,target)/ logit.size(0)
        #return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)
    def classification_loss(self, logit, target): #static
        return F.cross_entropy(logit,target) 
    def image_reconstruction_loss(self, Ia, Ia_hat, scalars):
        if self.hparams.rec_loss == 'l1':
            g_loss_rec = F.l1_loss(Ia,Ia_hat)
        elif self.hparams.rec_loss == 'l2':
            g_loss_rec = F.mse_loss(Ia,Ia_hat)
        elif self.hparams.rec_loss == 'perceptual':
            l1_loss=F.l1_loss(Ia,Ia_hat)
            scalars['G/loss_rec_l1'] = l1_loss.item()
            g_loss_rec = l1_loss
            #add perceptual loss
            f_img = self.vgg16_f(Ia)
            f_img_hat = self.vgg16_f(Ia_hat)
            if self.hparams.lambda_G_perc > 0:
                scalars['G/loss_rec_perc'] = self.hparams.lambda_G_perc * self.loss_P(f_img_hat, f_img)
                g_loss_rec += scalars['G/loss_rec_perc']
            if self.hparams.lambda_G_style > 0:
                scalars['G/loss_rec_style'] = self.hparams.lambda_G_style * self.loss_S(f_img_hat, f_img)
                g_loss_rec += scalars['G/loss_rec_style']
        return g_loss_rec

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

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

            # models parameters
            parser.add_argument('--g_conv_dim', default=32, type=int)
            parser.add_argument('--d_conv_dim', default=32, type=int)
            parser.add_argument('--g_layers', default=6, type=int)
            parser.add_argument('--d_layers', default=5, type=int)
            parser.add_argument('--d_fc_dim', default=512, type=int)
            parser.add_argument('--max_conv_dim', default=512, type=int)
            parser.add_argument('--skip_connections', default=0, type=int)
            parser.add_argument('--n_attr_deconv', default=1, type=int)#in how many deconv layer add the attribute:0 = no attribute, 1=normal (just concat to bneck)
            parser.add_argument('--vgg_like', default=0, type=int)

            #which part activate
            parser.add_argument('--rec_loss', default='l1', type=str)#Can be l1, l2, perceptual
            parser.add_argument('--use_image_disc', dest='use_image_disc', default=False, action='store_true')
            parser.add_argument('--use_latent_disc', dest='use_latent_disc', default=False, action='store_true')
            # optimizer parameters
            parser.add_argument('--batch_size', default=32, type=int)
            parser.add_argument('--optimizer', default='adam', type=str)
            parser.add_argument('--beta1', default=0.9, type=float)
            parser.add_argument('--beta2', default=0.999, type=float)
            parser.add_argument('--n_critic', default=1, type=int)
            parser.add_argument('--n_critic_ld', default=1, type=int)
            parser.add_argument('--g_lr', default=0.0002, type=float)
            parser.add_argument('--d_lr', default=0.0002, type=float)
            parser.add_argument('--ld_lr', default=0.00002, type=float)
            #weights of the losses
            parser.add_argument('--lambda_adv', default=1, type=float)
            parser.add_argument('--lambda_gp', default=10, type=float)
            parser.add_argument('--lambda_G_rec', default=1, type=float)
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
