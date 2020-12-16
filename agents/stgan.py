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
from PIL import ImageFont
from PIL import ImageDraw 
from torchvision import transforms
import numpy as np
import cv2
from sklearn.decomposition import PCA, FastICA
from loss.loss_provider import LossProvider
import torchvision.utils as tvutils

from datasets import *
from models.stgan import Generator, Discriminator, Latent_Discriminator, GeneratorWithBranches
#from models.vggPerceptualLoss import VGGPerceptualLoss
from models.perceptual_loss import PerceptualLoss, GradientL1Loss
from utils.misc import print_cuda_statistics
from utils.im_util import _imscatter
import matplotlib.pyplot as plt
import numpy as np



class STGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)

        if self.config.use_branches:
            self.G = GeneratorWithBranches(len(self.config.attrs),self.config.g_conv_dim, self.config.g_layers, self.config.max_conv_dim, self.config.shortcut_layers,n_attr_deconv=self.config.n_attr_deconv, vgg_like=self.config.vgg_like)
        else:
            self.G = Generator(len(self.config.attrs),self.config.g_conv_dim, self.config.g_layers, self.config.max_conv_dim, self.config.shortcut_layers,n_attr_deconv=self.config.n_attr_deconv, vgg_like=self.config.vgg_like)
        self.D = Discriminator(self.config.image_size, self.config.max_conv_dim, len(self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)
        self.LD = Latent_Discriminator(self.config.image_size, self.config.max_conv_dim, len(self.config.attrs), self.config.g_conv_dim, self.config.d_fc_dim, self.config.g_layers, 0)

        print(self.G)
        if self.config.use_image_disc or self.config.use_classifier_generator:
            print(self.D)
        if self.config.use_latent_disc:
            print(self.LD)

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
        if (self.config.use_image_disc or self.config.use_classifier_generator):
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
        if (self.config.use_image_disc or self.config.use_classifier_generator):
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
        self.load_checkpoint() #load checkpoint if needed and LOAD CLASSIFIERS
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.config.use_image_disc or self.config.use_classifier_generator))
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
            g_loss_rec = torch.mean(torch.abs(Ia - Ia_hat))
        elif self.config.rec_loss == 'l2':
            g_loss_rec = ((Ia - Ia_hat) ** 2).mean()
        elif self.config.rec_loss == 'perceptual' or self.config.rec_loss == 'watson':
            perc_loss = self.perceptual_loss(Ia, Ia_hat) * self.config.perc_loss_rec_weight
            l1_loss=torch.mean(torch.abs(Ia - Ia_hat))
            scalars['G/loss_rec_l1'] = l1_loss.item()
            scalars['G/loss_rec_perc'] = perc_loss.item()
            g_loss_rec = perc_loss + l1_loss
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
            #alphas = np.linspace(5.0, 5.0, 10)
            #alphas = [torch.FloatTensor([alpha]) for alpha in alphas]
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
            if self.config.use_classifier_generator:
                _, out_att = self.D(fake_image.detach())

            #write target and predicted score
            for im in range(fake_image.shape[0]):
                #image=(fake_image[im].cpu().detach().numpy().transpose((1,2,0))*127.5+127.5).astype(np.uint8)
                image=np.zeros((128,128,3), np.uint8)
                for i in range(c_trg_sample.shape[1]):
                    cv2.putText(image, "%.2f"%(c_trg_sample[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
                    if self.config.use_classifier_generator:
                        cv2.putText(image, "%.2f"%(out_att[im][i].item()), (10,14*(i+7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
                image=((image.astype(np.float32))/255).transpose(2,0,1)+fake_image[im].cpu().detach().numpy()
                fake_image[im]=torch.from_numpy(image) #transforms.ToTensor()(image)*(2)-1

            x_fake_list.append(fake_image)
        x_concat = torch.cat(x_fake_list, dim=3)
        if writer:
            self.writer.add_image('sample', make_grid(self.denorm(x_concat.data.cpu()), nrow=1),
                                    self.current_iteration)
        save_image(self.denorm(x_concat.data.cpu()),path,
                    nrow=1, padding=0)

    def compute_disentangle_grid(self,x_real, a_att, path,writer=False):
        #encode all the batch
        encodings,bneck = self.G.encode(x_real)
        splitted_bneck = self.G.split_bneck(bneck)
        batch_size=x_real.shape[0]
        
        #for each branch
        for label in range(len(splitted_bneck)):
            x_fake_list = [torch.cat((x_real[0].unsqueeze(0),x_real),dim=0)]
            #each column: all share the same embedding for the branchfor c in range(batch_size):
            for c in range(batch_size):
                #copy bneck and encodings (if branches)
                bneck_copy = [enc.clone() for enc in splitted_bneck]
                if self.config.use_branches:
                    enc_copy= [[enc.clone() for enc in encs] for encs in encodings]
                else:
                    enc_copy = encodings
                #change bneck for all images (and encodings if branches)
                for i in range(batch_size): #for each image
                    bneck_copy[label][i]=bneck_copy[label][c]
                    if self.config.use_branches:
                        for layer in range(len(enc_copy[label])):
                            enc_copy[label][layer][i]=enc_copy[label][layer][c]
                #decode for the encodings in bneck_copy
                bneck_copy = self.G.join_bneck(bneck_copy)
                #print(len(bneck), [bn.shape() for bn in bneck])
                fake_image=self.G.decode(bneck_copy,a_att,enc_copy)
                #add reference image
                fake_image=torch.cat((x_real[c].unsqueeze(0),fake_image),dim=0)
                x_fake_list.append(fake_image)
            x_concat = torch.cat(x_fake_list, dim=3)
            save_image(self.denorm(x_concat.data.cpu()),path.format(label),
                        nrow=1, padding=0)
            if writer:
                self.writer.add_image('disentangle_{}'.format(label), make_grid(self.denorm(x_concat.data.cpu()), nrow=1),
                                        self.current_iteration)
    def log_img_interpolation(self, img, label, mode):
        # forward through the model and get loss
        batch_size = img.size(0)
        enc_feat,bneck = self.G.encode(img)
        bneck_mater, bneck_shape, bneck_illum = self.G.split_bneck(bneck)
        bneck_size = bneck.shape

        all_recon = []
        for _ in range(batch_size):
            bneck = [bneck_mater, bneck_shape, bneck_illum]
            # only MATERIAL changes in the batch
            if torch.all(mode == 0):
                bneck_material = torch.roll(bneck_material, 1, dims=0)
            # only GEOMETRY changes in the batch
            elif torch.all(mode == 1):
                bneck_shape = torch.roll(bneck_shape, 1, dims=0)
            # only ILLUMINATION changes in the batch
            elif torch.all(mode == 2):
                bneck_illum = torch.roll(bneck_illum, 1, dims=0)

            bneck = self.G.join_bneck(bneck, bneck_size)
            img_hat = self.G.decode(bneck,label,enc_feat)
            all_recon.append(img_hat)

        all_recon = torch.cat((img, *all_recon), dim=-2)
        img_log = tvutils.make_grid(all_recon * 0.5 + 0.5, nrow=batch_size)
        self.writer.add_image('roll', img_log, self.current_iteration)

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




    def train(self):
        self.setup_all_optimizers()

        # samples used for testing (linear samples) the net
        val_iter = iter(self.data_loader.val_loader)
        Ia_sample, a_sample = next(val_iter)
        Ia_sample = Ia_sample.to(self.device)
        a_sample = a_sample.to(self.device)
        b_samples = self.create_labels(a_sample, self.config.attrs)
        b_samples.insert(0, a_sample.to(self.device))  # reconstruction


        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()

        start_batch = self.current_iteration // self.data_loader.train_iterations
        print(self.current_iteration,self.data_loader.train_iterations,start_batch)

        if self.config.rec_loss == 'perceptual':
            self.perceptual_loss = PerceptualLoss(use_gram_matrix=False).to(self.device)
            #perceptual_loss = GradientL1Loss().to(self.device)
        if self.config.rec_loss == 'watson':
            provider = LossProvider()
            self.perceptual_loss = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum').to(self.device)

        for batch in range(start_batch, self.config.max_epoch):
            for it in range(self.data_loader.train_iterations):
                # Ia_sample, a_sample, mode = next(val_iter)
                # Ia_sample = Ia_sample.to(self.device)
                # a_sample = a_sample.to(self.device)
                # b_samples = self.create_labels(a_sample, self.config.attrs)
                # b_samples.insert(0, a_sample.to(self.device))  # reconstruction

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                # fetch real images and labels
                try:
                    Ia, a_att, mode = next(data_iter)
                except:
                    data_iter = iter(self.data_loader.train_loader)
                    Ia, a_att, mode = next(data_iter)

                # generate target domain labels randomly
                b_att =  torch.rand_like(a_att)*2-1.0 # a_att + torch.randn_like(a_att)*self.config.gaussian_stddev

                Ia = Ia.to(self.device)         # input images
                a_att = a_att.to(self.device)   # labels for computing classification loss
                b_att = b_att.to(self.device)   # labels for computing classification loss

                # attr_diff = b_att - a_att #b_att_copy - a_att_copy
                # attr_diff = attr_diff if self.config.use_attr_diff else b_att
                scalars = {}


                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                if self.config.use_image_disc or self.config.use_classifier_generator:
                    self.G.eval()
                    self.D.train()

                    for _ in range(self.config.n_critic):
                        # input is the real image Ia
                        out_disc_real, out_att_real = self.D(Ia)

                        d_loss = 0
                        if self.config.use_image_disc:
                            # fake image Ib_hat
                            Ib_hat = self.G(Ia, b_att)
                            out_disc_fake, _ = self.D(Ib_hat.detach())
                            #adversarial losses
                            d_loss_adv_real = - torch.mean(out_disc_real)
                            d_loss_adv_fake = torch.mean(out_disc_fake)
                            # compute loss for gradient penalty
                            alpha = torch.rand(Ia.size(0), 1, 1, 1).to(self.device)
                            x_hat = (alpha * Ia.data + (1 - alpha) * Ib_hat.data).requires_grad_(True)
                            out_disc, _ = self.D(x_hat)
                            d_loss_adv_gp = self.gradient_penalty(out_disc, x_hat)
                            #full GAN loss
                            d_loss_adv = d_loss_adv_real + d_loss_adv_fake + self.config.lambda_gp * d_loss_adv_gp
                            d_loss += self.config.lambda_adv * d_loss_adv
                            scalars['D/loss_adv'] = d_loss_adv.item()
                            scalars['D/loss_real'] = d_loss_adv_real.item()
                            scalars['D/loss_fake'] = d_loss_adv_fake.item()
                            scalars['D/loss_gp'] = d_loss_adv_gp.item()

                        if self.config.use_classifier_generator:
                            d_loss_att = self.regression_loss(out_att_real, a_att)
                            d_loss += self.config.lambda_d_att * d_loss_att
                            #train with generator images
                            # if(self.current_iteration>30000):
                            #     Ia_hat = self.G(Ia, a_att)
                            #     _, out_att_fake = self.D(Ia_hat.detach())
                            #     d_loss_att2 = self.classification_loss(out_att_fake, a_att)
                            #     scalars['D/loss_att2'] = d_loss_att2.item()
                            #     d_loss += self.config.lambda_d_att * d_loss_att2
                            scalars['D/loss_att'] = d_loss_att.item()

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
                    bneck_mater, bneck_shape, bneck_illum = self.G.split_bneck(bneck)

                    for _ in range(self.config.n_critic_ld):
                        out_att = self.LD(bneck_mater)
                        #classification loss
                        ld_loss = self.regression_loss(out_att, a_att)*self.config.lambda_ld
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

                bneck_mater, bneck_shape, bneck_illum = self.G.split_bneck(bneck)
                bneck = [bneck_mater, bneck_shape, bneck_illum]
                # Compute invariancy losses
                loss_shape = 0
                loss_illum = 0
                loss_material = 0
                # clamp features to be equal to the batch mean and compute loss
                if torch.all(mode == 0):  # only MATERIAL changes in the batch
                    bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
                    bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
                    if self.config.do_mean_features : bneck = [bneck_mater, bneck_shape_mean, bneck_illum_mean]
                    loss_shape = torch.dist(bneck_shape,bneck_shape_mean, p=2).mean()
                    loss_illum = torch.dist(bneck_illum,bneck_illum_mean, p=2).mean()
                elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
                    bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
                    bneck_illum_mean = bneck_illum.mean(dim=0, keepdim=True).expand_as(bneck_illum)
                    if self.config.do_mean_features : bneck = [bneck_mater_mean, bneck_shape, bneck_illum_mean]
                    loss_illum = torch.dist(bneck_illum,bneck_illum_mean, p=2).mean()
                    loss_material = torch.dist(bneck_mater,bneck_mater_mean, p=2).mean()
                elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
                    bneck_shape_mean = bneck_shape.mean(dim=0, keepdim=True).expand_as(bneck_shape)
                    bneck_mater_mean = bneck_mater.mean(dim=0, keepdim=True).expand_as(bneck_mater)
                    if self.config.do_mean_features : bneck = [bneck_mater_mean, bneck_shape_mean, bneck_illum]
                    loss_shape = torch.dist(bneck_shape,bneck_shape_mean, p=2).mean()
                    loss_material = torch.dist(bneck_mater, bneck_mater_mean,p=2).mean()
                else:
                    raise ValueError('data sampling  mode not understood')
                scalars["G/dist_shape"]=loss_shape
                scalars["G/dist_illum"]=loss_illum
                scalars["G/dist_material"]=loss_material

                # join bneck
                bneck = self.G.join_bneck(bneck)

                Ia_hat=self.G.decode(bneck,a_att,encodings)
                g_loss_rec = self.reconstruction_loss(Ia,Ia_hat,scalars)
                g_loss = self.config.lambda_g_rec * g_loss_rec
        
                if self.config.lambda_G_features > 0:
                    invariancy_loss=(loss_shape + loss_illum + loss_material)*self.config.lambda_G_features
                    g_loss += invariancy_loss
                    scalars['G/loss_invariancy'] = invariancy_loss

                #latent discriminator for attribute in the material part TODO mat part only
                if self.config.use_latent_disc:
                    out_att = self.LD(bneck_mater)
                    g_loss_latent = -self.regression_loss(out_att, a_att)
                    g_loss += self.config.lambda_g_latent * g_loss_latent
                    scalars['G/loss_latent'] = g_loss_latent.item()

                if (self.config.use_image_disc or self.config.use_classifier_generator):
                    # original-to-target domain : Ib_hat -> GAN + classif
                    Ib_hat = self.G(Ia, b_att)
                    out_disc, out_att = self.D(Ib_hat)
                    # GAN loss
                    if self.config.use_image_disc:                    
                        g_loss_adv = - torch.mean(out_disc)
                        g_loss += self.config.lambda_adv * g_loss_adv
                        scalars['G/loss_adv'] = g_loss_adv.item()
                    #output classifier loss
                    if self.config.use_classifier_generator:
                        lambda_new = self.config.lambda_g_att * max(min((self.current_iteration-20000)/20000,1),0)
                        g_loss_att = self.regression_loss(out_att, b_att)
                        g_loss += lambda_new * g_loss_att
                        scalars['G/loss_att'] = g_loss_att.item()
                        scalars['G/lambda_new'] = lambda_new

                # backward and optimize
                self.optimize(self.optimizer_G,g_loss)
                # summarize
                scalars['G/loss_rec'] = g_loss_rec.item()
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
                        #path=os.path.join(self.config.sample_dir, 'disentangle_{}_{}.jpg'.format(self.current_iteration,"{}"))
                        #self.compute_disentangle_grid(Ia_sample,a_sample,path,writer=True)
                        self.compute_disentangle_grid(Ia_sample,a_sample,os.path.join(self.config.sample_dir, 'disentangle_{}_{}.jpg'.format(self.current_iteration,"{}")),writer=True)
                        self.log_img_interpolation(Ia,a_att,mode)
                # save checkpoint
                if self.current_iteration % self.config.checkpoint_step == 0:
                    self.save_checkpoint()



    def test_classif(self):
        self.load_checkpoint()
        [C.eval() for C in self.Cs]

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        self.D.eval()
        
        label_names={'shape':['blob', 'bunny', 'dragon', 'dragon2', 'einstein', 'einstein2', 'lucy', 'sphere', 'statue', 'suzanne', 'teapot', 'waterpot', 'zenith'],'illum':['doge2', 'ennis', 'glacier', 'grace', 'pisa', 'uffizi']}

        colors=['black','red','cyan','blue','green','yellow','grey','white', 'orange','blue','magenta','red', 'yellow']
        with torch.no_grad():
            #do it for shape and for mat
            for id_att,att in enumerate(label_names):
                all_figs=[]; ax=[]
                n_class = len(label_names[att])
                #print(id_att,att,n_class)
                print("testing for attribute",att,"with n_class=",n_class)
                for i in range(n_class):
                    all_figs.append(plt.figure(figsize=(12,12)))
                    ax.append(all_figs[i].gca())
                for i, (x_real, c_org,labels) in enumerate(tqdm_loader):
                    x_real = x_real.to(self.device)
                    _,pred = self.Cs[id_att](x_real)
                    pred = F.softmax(pred)
                    #print(labels)
                    for j in range(x_real.shape[0]): #for each image
                        #print(colors[labels[id_att+1][j]],labels[id_att+1][j], label_names[att][labels[id_att+1][j]],filen[j])
                        for label in range(n_class):
                            #print(label,pred[j][label])
                            #print(labels[j][id_att+1])
                            _imscatter(pred[j][label].cpu()+(np.random.random()/10-0.05), np.random.random(),
                                    image=x_real[j].cpu(),
                                    color=colors[labels[id_att+1][j]], #TODO put color depending on label
                                    zoom=0.1,
                                    ax=ax[label])
                            
                            # _imscatter(out_att[j][label].cpu(), np.random.random(),
                            #         image=x_real[j].cpu(),
                            #         color='white',
                            #         zoom=0.1,
                            #         ax=ax[att][1])
                            #print(out_att[j][att])
                for label in range(n_class):
                    result_path = os.path.join(self.config.result_dir, 'scores_{}_{}_{}.jpg'.format(att,label_names[att][label],self.config.checkpoint))
                    #print(result_path)
                    all_figs[label].savefig(result_path, dpi=300, bbox_inches='tight')

                    #all_figs[i][0].show()
                    # result_path = os.path.join(self.config.result_dir, 'scores_{}_{}_{}.jpg'.format(label,1,self.config.checkpoint))
                    # all_figs[i][1].savefig(result_path, dpi=300, bbox_inches='tight')

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




    def test_disentangle(self):
        self.load_checkpoint()
        self.G.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        with torch.no_grad():
            for batch, (x_real, a_att) in enumerate(tqdm_loader): #MANU change to your outputs of the loader
                x_real=x_real.to(self.device)
                a_att=a_att.to(self.device)
                path=os.path.join(self.config.result_dir, 'sample_disentangle_{}_{}_{}.jpg'.format(batch + 1,"{}",self.config.checkpoint))
                self.compute_disentangle_grid(x_real,a_att,path)
                

  


    def test_pca(self):
        self.load_checkpoint()
        self.G.to(self.device)
        self.G.eval()

        # go through all training samples to get PCA dimensions
        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))
        with torch.no_grad():
            partial_embs = []
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                attr_diff = c_org-c_org if self.config.use_attr_diff else c_org #* self.config.thres_int
                _,encoded=self.G(x_real.to(self.device), attr_diff.to(self.device))
                partial_embs.append(encoded[-1].detach().cpu())
            embs = torch.cat(partial_embs, 0).numpy()
            embs=embs.reshape(embs.shape[0],-1)
        #fit PCA
        print(embs.shape)
        pca = PCA(n_components=100)  
        pca.fit(embs)
        reducted_trained_emb = pca.transform(embs)
        min_axis=np.min(reducted_trained_emb,axis=0)
        max_axis=np.max(reducted_trained_emb,axis=0)
        margin_axis=(max_axis-min_axis)*0.2
        # ratio_component = pca.explained_variance_ratio_
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        # print("20 first PCA componenents: ",ratio_component[0:20],
        #         " => {:0.4f}".format(sum(ratio_component[0:20])))

        #go through all batches and try to move in PCA space
        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                n_images=x_real.shape[0]
                attr_diff = c_org-c_org if self.config.use_attr_diff else c_org #* self.config.thres_int
                _,encoded=self.G(x_real.to(self.device), attr_diff.to(self.device))
                numpy_encoded=encoded[-1].detach().cpu().numpy()
                reducted_emb = pca.transform(numpy_encoded.reshape(n_images,-1))

                for axis in range (10):
                    #samples = reducted_emb[:,axis]+
                    samples=np.broadcast_to(np.linspace(min_axis[axis]-margin_axis[axis], max_axis[axis]+margin_axis[axis], 10), (n_images,10)).transpose(1,0)
                    x_fake_list = [x_real.to(self.device)]
                    for sample in samples:
                        edited_embs = np.copy(reducted_emb)
                        edited_embs[:, axis] = sample
                        edited_embs = pca.inverse_transform(edited_embs)
                        tensor_codes = torch.from_numpy(edited_embs.reshape(edited_embs.shape[0],512,2,2).astype(np.float32))
                        fake_image = self.G.decode(tensor_codes.to(self.device),attr_diff.to(self.device))
                        #edited_embs = np.repeat(reducted_emb, 10, 0)
                        x_fake_list.append(fake_image)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    save_image(self.denorm(x_concat.data.cpu()),os.path.join(self.config.result_dir, 'sample_pca_axis_{}_{}.jpg'.format(axis,i + 1)),nrow=1, padding=0)




    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
