import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import cv2
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGenerator, Discriminator, Latent_Discriminator
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.misc import print_cuda_statistics
import numpy as np



class FaderNet(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)

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

        self.current_iteration = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir)



    ################################################################
    ###################### SAVE/lOAD ###############################
    def save_checkpoint(self):
        def save_one_model(model,optimizer,name):
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,self.current_iteration)))

        save_one_model(self.G,self.optimizer_G,'G')
        if self.config.use_image_disc:
            save_one_model(self.D,self.optimizer_D,'D')
        if self.config.use_latent_disc:
            save_one_model(self.LD,self.optimizer_LD,'LD')

    def load_checkpoint(self):
        def load_one_model(model,optimizer,name, iter=self.config.checkpoint):
            G_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,iter)),map_location=self.device)
            G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
            model.load_state_dict(G_to_load)
            model.to(self.device)
            if optimizer != None:
                optimizer.load_state_dict(G_checkpoint['optimizer'])
                        
        if self.config.checkpoint is None:
            return

        load_one_model(self.G,self.optimizer_G if self.config.mode=='train' else None,'G')
        if (self.config.use_image_disc):
            load_one_model(self.D,self.optimizer_D if self.config.mode=='train' else None,'D')
        if self.config.use_latent_disc:
            load_one_model(self.LD,self.optimizer_LD if self.config.mode=='train' else None,'LD')

        self.current_iteration = self.config.checkpoint


    ################################################################
    ################### OPTIM UTILITIES ############################
    def build_optimizer(self,model,lr):
        model=model.to(self.device)
        return optim.Adam(model.parameters(), lr, [self.config.beta1, self.config.beta2])
    def build_scheduler(self,optimizer,not_load=False):
        last_epoch=-1 if (self.config.checkpoint == None or not_load) else self.config.checkpoint
        return optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_iters, gamma=0.1, last_epoch=last_epoch)
    def setup_all_optimizers(self):
        self.optimizer_G = self.build_optimizer(self.G, self.config.g_lr)
        self.optimizer_D = self.build_optimizer(self.D, self.config.d_lr)
        self.optimizer_LD = self.build_optimizer(self.LD, self.config.ld_lr)
        self.load_checkpoint() #load checkpoint if needed 
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.config.use_image_disc))
        self.lr_scheduler_LD = self.build_scheduler(self.optimizer_LD, not self.config.use_latent_disc)

    def optimize(self,optimizer,loss):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()




    ################################################################
    ################### LOSSES UTILITIES ###########################
    def regression_loss(self, logit, target):
        return F.l1_loss(logit,target)/ logit.size(0)
        #return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)
    def classification_loss(self, logit, target):
        return F.cross_entropy(logit,target) 
    def reconstruction_loss(self, Ia, Ia_hat, scalars):
        if self.config.rec_loss == 'l1':
            g_loss_rec = F.l1_loss(Ia,Ia_hat)
        elif self.config.rec_loss == 'l2':
            g_loss_rec = F.mse_loss(Ia,Ia_hat)
        elif self.config.rec_loss == 'perceptual':
            l1_loss=F.l1_loss(Ia,Ia_hat)
            scalars['G/loss_rec_l1'] = l1_loss.item()
            g_loss_rec = l1_loss
            #add perceptual loss
            f_img = self.vgg16_f(Ia)
            f_img_hat = self.vgg16_f(Ia_hat)
            if self.config.lambda_G_perc > 0:
                scalars['G/loss_rec_perc'] = self.config.lambda_G_perc * self.loss_P(f_img_hat, f_img)
                g_loss_rec += scalars['G/loss_rec_perc']
            if self.config.lambda_G_style > 0:
                scalars['G/loss_rec_style'] = self.config.lambda_G_style * self.loss_S(f_img_hat, f_img)
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



    ################################################################
    ##################### EVAL UTILITIES ###########################
    def denorm(self, x):
        #get from [-1,1] to [0,1]
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""
        c_trg_list = []
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def write_labels_on_images(self,images, labels):
        for im in range(images.shape[0]):
            text_image=np.zeros((128,128,3), np.uint8)
            for i in range(labels.shape[1]):
                cv2.putText(text_image, "%.2f"%(labels[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
            image_numpy=((text_image.astype(np.float32))/255).transpose(2,0,1)+images[im].cpu().detach().numpy()
            images[im]= torch.from_numpy(image_numpy)

    def compute_sample_grid(self,x_sample,c_sample_list,c_org_sample,path,writer=False):
        x_sample = x_sample.to(self.device)
        x_fake_list = [x_sample]
        for c_trg_sample in c_sample_list:
            fake_image=self.G(x_sample, c_trg_sample)
            self.write_labels_on_images(fake_image,c_trg_sample)
            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        if writer:
            self.writer.add_image('sample', make_grid(self.denorm(x_concat.data.cpu()), nrow=1),
                                    self.current_iteration)
        save_image(self.denorm(x_concat.data.cpu()),path,
                    nrow=1, padding=0)







    ################################################################
    ##################### MAIN FUNCTIONS ###########################
    def run(self):
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'train':
                self.train()
            else:
                #self.test_pca()
                self.test()
                self.test_disentangle()
                #self.test_classif()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
            traceback.print_exc()
        finally:
            self.finalize()





    ########################################################################################
    #####################                 TRAINING               ###########################
    def train(self):
        self.setup_all_optimizers()

        # samples used for testing (linear samples) the net
        val_iter = iter(self.data_loader.val_loader)
        Ia_sample, _, _, a_sample = next(val_iter)
        Ia_sample = Ia_sample.to(self.device)
        a_sample = a_sample.to(self.device)
        b_samples = self.create_labels(a_sample, self.config.attrs)
        b_samples.insert(0, a_sample.to(self.device))  # reconstruction


        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()

        start_batch = self.current_iteration // self.data_loader.train_iterations
        print(self.current_iteration,self.data_loader.train_iterations,start_batch)


        for batch in range(start_batch, self.config.max_epoch):
            for it in range(self.data_loader.train_iterations):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                # fetch real images and labels
                try:
                    Ia, _, _, a_att = next(data_iter)
                except:
                    data_iter = iter(self.data_loader.train_loader)
                    Ia, _, _, a_att = next(data_iter)

                # generate target domain labels randomly
                b_att =  torch.rand_like(a_att)*2-1.0 # a_att + torch.randn_like(a_att)*self.config.gaussian_stddev

                Ia = Ia.to(self.device)         # input images
                a_att = a_att.to(self.device)   # attribute of image
                b_att = b_att.to(self.device)   # fake attribute (if GAN/classifier)

                scalars = {}

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                if self.config.use_image_disc:
                    self.G.eval()
                    self.D.train()

                    for _ in range(self.config.n_critic):
                        # input is the real image Ia
                        out_disc_real = self.D(Ia)

                        d_loss = 0
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
                        d_loss_adv_gp = self.gradient_penalty(out_disc, x_hat)
                        #full GAN loss
                        d_loss_adv = d_loss_adv_real + d_loss_adv_fake + self.config.lambda_gp * d_loss_adv_gp
                        d_loss += self.config.lambda_adv * d_loss_adv
                        scalars['D/loss_adv'] = d_loss_adv.item()
                        scalars['D/loss_real'] = d_loss_adv_real.item()
                        scalars['D/loss_fake'] = d_loss_adv_fake.item()
                        scalars['D/loss_gp'] = d_loss_adv_gp.item()

                        # backward and optimize
                        self.optimize(self.optimizer_D,d_loss)
                        # summarize
                        scalars['D/loss'] = d_loss.item()


                # =================================================================================== #
                #                         3. Train the latent discriminator (FaderNet)                #
                # =================================================================================== #
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


                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                self.G.train()
                self.D.eval()
                self.LD.eval()

                encodings,bneck = self.G.encode(Ia)

                Ia_hat=self.G.decode(a_att,bneck,encodings)
                g_loss_rec = self.reconstruction_loss(Ia,Ia_hat,scalars)
                g_loss = self.config.lambda_G_rec * g_loss_rec
                scalars['G/loss_rec'] = g_loss_rec.item()
        

                #latent discriminator for attribute in the material part TODO mat part only
                if self.config.use_latent_disc:
                    out_att = self.LD(bneck_mater)
                    g_loss_latent = -self.regression_loss(out_att, a_att)
                    g_loss += self.config.lambda_G_latent * g_loss_latent
                    scalars['G/loss_latent'] = g_loss_latent.item()

                if self.config.use_image_disc:
                    # original-to-target domain : Ib_hat -> GAN + classif
                    Ib_hat = self.G(Ia, b_att)
                    out_disc = self.D(Ib_hat)
                    # GAN loss
                    g_loss_adv = - torch.mean(out_disc)
                    g_loss += self.config.lambda_adv * g_loss_adv
                    scalars['G/loss_adv'] = g_loss_adv.item()


                # backward and optimize
                self.optimize(self.optimizer_G,g_loss)
                # summarize
                scalars['G/loss'] = g_loss.item()

                self.current_iteration += 1

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #''
                self.lr_scheduler_G.step()
                self.lr_scheduler_D.step()
                self.lr_scheduler_LD.step()
                scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
                scalars['lr/ld_lr'] = self.lr_scheduler_LD.get_lr()[0]
                scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]

                # print summary on terminal and on tensorboard
                if self.current_iteration % self.config.summary_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('Elapsed [{}], Iteration [{}/{}] Epoch [{}/{}] (Iteration {})'.format(et, it, self.data_loader.train_iterations, batch, self.config.max_epoch,self.current_iteration))
                    #print(scalars)
                    for tag, value in scalars.items():
                        self.writer.add_scalar(tag, value, self.current_iteration)

                # sample
                if (self.current_iteration) % self.config.sample_step == 0:
                    self.G.eval()
                    with torch.no_grad():
                        self.compute_sample_grid(Ia_sample,b_samples,a_sample,os.path.join(self.config.sample_dir, 'sample_{}.jpg'.format(self.current_iteration)),writer=True)
                # save checkpoint
                if self.current_iteration % self.config.checkpoint_step == 0:
                    self.save_checkpoint()





    ####################################################################################
    #####################                 TEST               ###########################

    def test(self):
        self.load_checkpoint()
        self.G.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                c_trg_list = self.create_labels(c_org, self.config.attrs,max_val=3.0)
                c_trg_list.insert(0, c_org)
                self.compute_sample_grid(x_real,c_trg_list,c_org,os.path.join(self.config.result_dir, 'sample_{}_{}.jpg'.format(i + 1,self.config.checkpoint)),writer=False)
                c_trg_list = self.create_labels(c_org, self.config.attrs,max_val=5.0)
                c_trg_list.insert(0, c_org)
                self.compute_sample_grid(x_real,c_trg_list,c_org,os.path.join(self.config.result_dir, 'sample_big_{}_{}.jpg'.format(i + 1,self.config.checkpoint)),writer=False)



    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
