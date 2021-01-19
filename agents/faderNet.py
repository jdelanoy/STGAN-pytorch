import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGenerator, Discriminator, Latent_Discriminator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.trainingModule import TrainingModule



class FaderNet(TrainingModule):
    def __init__(self, config):
        super(FaderNet, self).__init__(config)

        self.G = FaderNetGenerator(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv)
        self.D = Discriminator(image_size=config.image_size, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim)
        self.LD = Latent_Discriminator(image_size=config.image_size, attr_dim=len(config.attrs), conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like)


        print(self.G)
        if self.config.use_image_disc:
            print(self.D)
        if self.config.use_latent_disc:
            print(self.LD)

         # create all the loss functions that we may need for perceptual loss
        self.loss_P = PerceptualLoss()
        self.loss_S = StyleLoss()
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4'])


        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation)




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


    def create_labels(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""
        c_trg_list = [c_org]
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg.to(self.device))
        return c_trg_list



    def compute_sample_grid(self,x_sample,c_sample_list,c_org_sample,path,writer=False):
        x_sample = x_sample.to(self.device)
        x_fake_list = [x_sample]
        for c_trg_sample in c_sample_list:
            fake_image=self.G(x_sample, c_trg_sample)
            write_labels_on_images(fake_image,c_trg_sample)
            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        image = denorm(x_concat.data.cpu())
        if writer:
            self.writer.add_image('sample', make_grid(image, nrow=1),
                                    self.current_iteration)
        save_image(image,path,
                    nrow=1, padding=0)










    ########################################################################################
    #####################                 TRAINING               ###########################
    def training_step(self, batch):
        # ================================================================================= #
        #                            1. Preprocess input data                               #
        # ================================================================================= #
        Ia, _, _, a_att = batch
        # generate target domain labels randomly
        b_att =  torch.rand_like(a_att)*2-1.0 # a_att + torch.randn_like(a_att)*self.config.gaussian_stddev

        Ia = Ia.to(self.device)         # input images
        a_att = a_att.to(self.device)   # attribute of image
        b_att = b_att.to(self.device)   # fake attribute (if GAN/classifier)

        scalars = {}
        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.config.use_image_disc:
            self.G.eval()
            self.D.train()

            for _ in range(self.config.n_critic):
                # input is the real image Ia
                out_disc_real = self.D(Ia)
                # fake image Ib_hat
                Ib_hat = self.G(Ia, b_att)
                out_disc_fake = self.D(Ib_hat.detach())
                #adversarial losses
                d_loss_adv_real = - torch.mean(out_disc_real)
                d_loss_adv_fake = torch.mean(out_disc_fake)
                # compute loss for gradient penalty
                alpha = torch.rand(Ia.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * Ia.data + (1 - alpha) * Ib_hat.data).requires_grad_(True)
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

        Ia_hat=self.G.decode(a_att,bneck,encodings)
        g_loss_rec = self.config.lambda_G_rec * self.image_reconstruction_loss(Ia,Ia_hat,scalars)
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
            Ib_hat = self.G(Ia, b_att)
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
        Ia_sample, _, _, a_sample = batch
        Ia_sample = Ia_sample.to(self.device)
        a_sample = a_sample.to(self.device)
        b_samples = self.create_labels(a_sample, self.config.attrs)
        self.compute_sample_grid(Ia_sample,b_samples,a_sample,os.path.join(self.config.sample_dir, 'sample_{}.jpg'.format(self.current_iteration)),writer=True)


    def testing_step(self, batch, batch_id):
        i, (x_real, _, _, c_org) = batch_id, batch
        c_trg_list = self.create_labels(c_org, self.config.attrs,max_val=3.0)
        self.compute_sample_grid(x_real,c_trg_list,c_org,os.path.join(self.config.result_dir, 'sample_{}_{}.jpg'.format(i + 1,self.config.checkpoint)),writer=False)
        c_trg_list = self.create_labels(c_org, self.config.attrs,max_val=5.0)
        self.compute_sample_grid(x_real,c_trg_list,c_org,os.path.join(self.config.result_dir, 'sample_big_{}_{}.jpg'.format(i + 1,self.config.checkpoint)),writer=False)


