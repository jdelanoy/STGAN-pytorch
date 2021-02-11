import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGeneratorWithNormals, Discriminator, Latent_Discriminator, Unet
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.trainingModule import TrainingModule



class FaderNetWithNormals(TrainingModule):
    def __init__(self, config):
        super(FaderNetWithNormals, self).__init__(config)

        self.G = FaderNetGeneratorWithNormals(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals)
        self.D = Discriminator(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim)
        self.LD = Latent_Discriminator(image_size=config.image_size, im_channels=config.img_channels, attr_dim=len(config.attrs), conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like)

        ### load the normal predictor network
        self.normal_G = Unet(conv_dim=config.g_conv_dim_normals,n_layers=config.g_layers_normals,max_dim=config.max_conv_dim_normals, im_channels=config.img_channels, skip_connections=config.skip_connections_normals, vgg_like=config.vgg_like_normals)
        self.load_model_from_path(self.normal_G,config.normal_predictor_checkpoint)
        self.normal_G.eval()


        print(self.G)
        if self.config.use_image_disc:
            print(self.D)
        if self.config.use_latent_disc:
            print(self.LD)

         # create all the loss functions that we may need for perceptual loss
        self.loss_P = PerceptualLoss().to(self.device)
        self.loss_S = StyleLoss().to(self.device)
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4']).to(self.device)


        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg)

        self.logger.info("FaderNet with normals ready")



    ################################################################
    ###################### SAVE/lOAD ###############################
    def save_checkpoint(self):
        self.save_one_model(self.G,self.optimizer_G,'G')
        if self.config.use_image_disc:
            self.save_one_model(self.D,self.optimizer_D,'D')
        if self.config.use_latent_disc:
            self.save_one_model(self.LD,self.optimizer_LD,'LD')

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            return

        self.load_one_model(self.G,self.optimizer_G if self.config.mode=='train' else None,'G')
        if (self.config.use_image_disc):
            self.load_one_model(self.D,self.optimizer_D if self.config.mode=='train' else None,'D')
        if self.config.use_latent_disc:
            self.load_one_model(self.LD,self.optimizer_LD if self.config.mode=='train' else None,'LD')

        self.current_iteration = self.config.checkpoint


    ################################################################
    ################### OPTIM UTILITIES ############################

    def setup_all_optimizers(self):
        self.optimizer_G = self.build_optimizer(self.G, self.config.g_lr)
        self.optimizer_D = self.build_optimizer(self.D, self.config.d_lr)
        self.optimizer_LD = self.build_optimizer(self.LD, self.config.ld_lr)
        self.load_checkpoint() #load checkpoint if needed 
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.config.use_image_disc))
        self.lr_scheduler_LD = self.build_scheduler(self.optimizer_LD, not self.config.use_latent_disc)

    def step_schedulers(self,scalars):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
        self.lr_scheduler_LD.step()
        scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
        scalars['lr/ld_lr'] = self.lr_scheduler_LD.get_lr()[0]
        scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]

    def eval_mode(self):
        self.G.eval()
        self.LD.eval()
        self.D.eval()
    def training_mode(self):
        self.G.train()
        self.LD.train()
        self.D.train()



    ################################################################
    ##################### EVAL UTILITIES ###########################


    def create_interpolated_attr(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""
        c_trg_list = [c_org.to(self.device)]
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg.to(self.device))
        return c_trg_list



    def compute_sample_grid(self,batch,max_val,path=None,writer=False):
        x_sample, normals, _, c_org_sample = batch
        c_sample_list = self.create_interpolated_attr(c_org_sample, self.config.attrs,max_val=max_val)
        x_sample = x_sample.to(self.device)
        c_org_sample = c_org_sample.to(self.device)
        normals=normals[:,:3].to(self.device) #self.get_normals(x_sample)

        x_fake_list = [normals,x_sample[:,:3]]
        for c_trg_sample in c_sample_list:
            fake_image=self.G(x_sample, c_trg_sample,normals)*x_sample[:,3:]
            write_labels_on_images(fake_image,c_trg_sample)
            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(denorm(x_concat), nrow=1)
        if writer:
            self.writer.add_image('sample', image,self.current_iteration)
        if path:
            tvutils.save_image(image,path)

    def get_normals(self, img):
        normals=self.normal_G(img)
        return normals*img[:,3:]








    ########################################################################################
    #####################                 TRAINING               ###########################
    def training_step(self, batch):
        # ================================================================================= #
        #                            1. Preprocess input data                               #
        # ================================================================================= #
        Ia, normals, _, a_att = batch
        # generate target domain labels randomly
        b_att =  torch.rand_like(a_att)*2-1.0 # a_att + torch.randn_like(a_att)*self.config.gaussian_stddev

        Ia = Ia.to(self.device)         # input images
        Ia_3ch = Ia[:,:3]
        mask = Ia[:,3:]
        a_att = a_att.to(self.device)   # attribute of image
        b_att = b_att.to(self.device)   # fake attribute (if GAN/classifier)
        normals_hat=normals[:,:3].to(self.device) #self.get_normals(Ia)

        scalars = {}
        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.config.use_image_disc:
            self.G.eval()
            self.D.train()

            for _ in range(self.config.n_critic):
                # input is the real image Ia
                out_disc_real = self.D(Ia_3ch)
                # fake image Ib_hat
                Ib_hat = self.G(Ia, b_att, normals_hat)
                out_disc_fake = self.D(Ib_hat.detach())
                #adversarial losses
                d_loss_adv_real = - torch.mean(out_disc_real)
                d_loss_adv_fake = torch.mean(out_disc_fake)
                # compute loss for gradient penalty
                alpha = torch.rand(Ia.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * Ia_3ch.data + (1 - alpha) * Ib_hat.data).requires_grad_(True)
                out_disc = self.D(x_hat)
                d_loss_adv_gp = self.config.lambda_gp * self.gradient_penalty(out_disc, x_hat)
                #full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss = self.config.lambda_adv * d_loss_adv
                scalars['D/loss_adv'] = d_loss.item()
                scalars['D/loss_real'] = d_loss_adv_real.item()
                scalars['D/loss_fake'] = d_loss_adv_fake.item()
                scalars['D/loss_gp'] = d_loss_adv_gp.item()

                # backward and optimize
                self.optimize(self.optimizer_D,d_loss)
                # summarize
                scalars['D/loss'] = d_loss.item()


        # ================================================================================= #
        #                        3. Train the latent discriminator (FaderNet)               #
        # ================================================================================= #
        if self.config.use_latent_disc:
            self.G.eval()
            self.LD.train()
            # compute disc loss on encoded image
            _,bneck = self.G.encode(Ia)

            for _ in range(self.config.n_critic_ld):
                out_att = self.LD(bneck)
                #classification loss
                ld_loss = self.regression_loss(out_att, a_att)*self.config.lambda_LD
                # backward and optimize
                self.optimize(self.optimizer_LD,ld_loss)
                # summarize
                scalars['LD/loss'] = ld_loss.item()


        # ================================================================================= #
        #                              3. Train the generator                               #
        # ================================================================================= #
        self.G.train()
        self.D.eval()
        self.LD.eval()

        encodings,bneck = self.G.encode(Ia)

        Ia_hat=self.G.decode(a_att,bneck, normals_hat,encodings)
        g_loss_rec = self.config.lambda_G_rec * self.image_reconstruction_loss(Ia_3ch,Ia_hat,scalars)
        g_loss = g_loss_rec
        scalars['G/loss_rec'] = g_loss_rec.item()

        #latent discriminator for attribute in the material part TODO mat part only
        if self.config.use_latent_disc:
            out_att = self.LD(bneck)
            g_loss_latent = -self.config.lambda_G_latent * self.regression_loss(out_att, a_att)
            g_loss += g_loss_latent
            scalars['G/loss_latent'] = g_loss_latent.item()

        if self.config.use_image_disc:
            # original-to-target domain : Ib_hat -> GAN + classif
            Ib_hat = self.G(Ia, b_att, normals_hat)
            out_disc = self.D(Ib_hat)
            # GAN loss
            g_loss_adv = - self.config.lambda_adv * torch.mean(out_disc)
            g_loss += g_loss_adv
            scalars['G/loss_adv'] = g_loss_adv.item()

        # backward and optimize
        self.optimize(self.optimizer_G,g_loss)
        # summarize
        scalars['G/loss'] = g_loss.item()





        return scalars

    def validating_step(self, batch):
        self.compute_sample_grid(batch,5.0,os.path.join(self.config.sample_dir, 'sample_{}.png'.format(self.current_iteration)),writer=True)


    def testing_step(self, batch, batch_id):
        i=batch_id
        self.compute_sample_grid(batch,3.0,os.path.join(self.config.result_dir, 'sample_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)
        self.compute_sample_grid(batch,5.0,os.path.join(self.config.result_dir, 'sample_big_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)


