import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGenerator, Discriminator, Latent_Discriminator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor
from modules.GAN_loss import GANLoss
from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.trainingModule import TrainingModule



class FaderNet(TrainingModule):
    def __init__(self, config):
        super(FaderNet, self).__init__(config)

        self.G = FaderNetGenerator(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv)
        self.D = Discriminator(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim)
        self.LD = Latent_Discriminator(image_size=config.image_size, im_channels=config.img_channels, attr_dim=len(config.attrs), conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like)

        print(self.G)
        if self.config.use_image_disc:
            print(self.D)
        if self.config.use_latent_disc:
            print(self.LD)

         # create all the loss functions that we may need for perceptual loss
        self.loss_P = PerceptualLoss().to(self.device)
        self.loss_S = StyleLoss().to(self.device)
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4']).to(self.device)
        self.criterionGAN = GANLoss(self.config.gan_mode).to(self.device)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg)

        self.logger.info("FaderNet ready")



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

    def step_schedulers(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
        self.lr_scheduler_LD.step()
        self.scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
        self.scalars['lr/ld_lr'] = self.lr_scheduler_LD.get_lr()[0]
        self.scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]

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
    def decode(self,bneck,encodings,att):
        return self.G.decode(att,bneck,encodings)
    def encode(self):
        return self.G.encode(self.batch_Ia)
    def forward(self,new_attr=None):
        encodings,z = self.encode()
        if new_attr != None: att=new_attr
        else: att = self.batch_a_att
        return self.decode(z,encodings,att)



    def init_sample_grid(self):
        x_fake_list = [self.batch_Ia[:,:3]]
        return x_fake_list

    def create_interpolated_attr(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""
        c_trg_list = [c_org]
        for i in range(len(selected_attrs)):
            alphas = [-max_val, -((max_val-1)/2.0+1), -1,-0.5,0,0.5,1,((max_val-1)/2.0+1), max_val]
            #alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg)
        return c_trg_list



    def compute_sample_grid(self,batch,max_val,path=None,writer=False):
        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch
        c_sample_list = self.create_interpolated_attr(self.batch_a_att, self.config.attrs,max_val=max_val)

        x_fake_list = self.init_sample_grid()
        for c_trg_sample in c_sample_list:
            fake_image=self.forward(c_trg_sample)*self.batch_Ia[:,3:]
            write_labels_on_images(fake_image,c_trg_sample)
            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(denorm(x_concat), nrow=1)
        if writer:
            self.writer.add_image('sample', image,self.current_iteration)
        if path:
            tvutils.save_image(image,path)










    ########################################################################################
    #####################                 TRAINING               ###########################
    def train_latent_discriminator(self):
        self.G.eval()
        self.LD.train()
        # compute disc loss on encoded image
        _,bneck = self.encode()

        for _ in range(self.config.n_critic_ld):
            out_att = self.LD(bneck)
            #classification loss
            ld_loss = self.regression_loss(out_att, self.batch_a_att)*self.config.lambda_LD
            # backward and optimize
            self.optimize(self.optimizer_LD,ld_loss)
            # summarize
            self.scalars['LD/loss'] = ld_loss.item()



    def train_GAN_discriminator(self):
        self.G.eval()
        self.D.train()

        for _ in range(self.config.n_critic):
            b_att =  torch.rand_like(self.batch_a_att)*2-1.0 
            # input is the real image Ia
            out_disc_real = self.D(self.batch_Ia[:,:3])
            # fake image Ib_hat
            Ib_hat = self.forward(b_att) #G(Ia, b_att)
            out_disc_fake = self.D(Ib_hat.detach())
            #adversarial losses
            d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
            d_loss_adv_real = self.criterionGAN(out_disc_real, True)
            # d_loss_adv_real = - torch.mean(out_disc_real)
            # d_loss_adv_fake = torch.mean(out_disc_fake)
            # compute loss for gradient penalty
            d_loss_adv_gp = GANLoss.cal_gradient_penalty(self.D,self.batch_Ia[:,:3],Ib_hat,self.device,lambda_gp=self.config.lambda_gp)
            # alpha = torch.rand(Ia.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * Ia[:,:3].data + (1 - alpha) * Ib_hat.data).requires_grad_(True)
            # out_disc = self.D(x_hat)
            # d_loss_adv_gp = self.config.lambda_gp * self.gradient_penalty(out_disc, x_hat)
            #full GAN loss
            d_loss = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
            self.scalars['D/loss_real'] = d_loss_adv_real.item()
            self.scalars['D/loss_fake'] = d_loss_adv_fake.item()
            self.scalars['D/loss_gp'] = d_loss_adv_gp.item()

            # backward and optimize
            self.optimize(self.optimizer_D,d_loss)
            # summarize
            self.scalars['D/loss'] = d_loss.item()




    def training_step(self, batch):
        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch

        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.config.use_image_disc:
            self.train_GAN_discriminator()

        # ================================================================================= #
        #                        3. Train the latent discriminator (FaderNet)               #
        # ================================================================================= #
        if self.config.use_latent_disc:
            self.train_latent_discriminator()

        # ================================================================================= #
        #                              3. Train the generator                               #
        # ================================================================================= #
        self.G.train()
        self.D.eval()
        self.LD.eval()

        encodings,bneck = self.encode()
        Ia_hat=self.decode(bneck,encodings,self.batch_a_att)

        #reconstruction loss
        g_loss_rec = self.config.lambda_G_rec * self.image_reconstruction_loss(self.batch_Ia[:,:3],Ia_hat)
        g_loss = g_loss_rec
        self.scalars['G/loss_rec'] = g_loss_rec.item()

        #latent discriminator for attribute
        if self.config.use_latent_disc:
            out_att = self.LD(bneck)
            g_loss_latent = -self.config.lambda_G_latent * self.regression_loss(out_att, self.batch_a_att)
            g_loss += g_loss_latent
            self.scalars['G/loss_latent'] = g_loss_latent.item()

        if self.config.use_image_disc:
            b_att =  torch.rand_like(self.batch_a_att)*2-1.0 
            # original-to-target domain : Ib_hat -> GAN + classif
            Ib_hat = self.forward(b_att)
            out_disc = self.D(Ib_hat)
            # GAN loss
            g_loss_adv = self.config.lambda_adv * self.criterionGAN(out_disc, True)
            #g_loss_adv = - self.config.lambda_adv * torch.mean(out_disc)
            g_loss += g_loss_adv
            self.scalars['G/loss_adv'] = g_loss_adv.item()

        # backward and optimize
        self.optimize(self.optimizer_G,g_loss)
        # summarize
        self.scalars['G/loss'] = g_loss.item()
        return self.scalars

    def validating_step(self, batch):
        self.compute_sample_grid(batch,5.0,os.path.join(self.config.sample_dir, 'sample_{}.png'.format(self.current_iteration)),writer=True)


    def testing_step(self, batch, batch_id):
        i=batch_id
        self.compute_sample_grid(batch,3.0,os.path.join(self.config.result_dir, 'sample_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)
        self.compute_sample_grid(batch,5.0,os.path.join(self.config.result_dir, 'sample_big_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)



