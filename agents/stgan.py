import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import ImageFont
from PIL import ImageDraw 
from torchvision import transforms
import numpy as np
import cv2
from sklearn.decomposition import PCA, FastICA

from datasets import *
from models.stgan import Generator, Discriminator, Latent_Discriminator
from utils.misc import print_cuda_statistics
from utils.im_util import _imscatter
import matplotlib.pyplot as plt
import numpy as np

cudnn.benchmark = True


class STGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        
        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers, self.config.max_conv_dim, self.config.shortcut_layers, use_stu=self.config.use_stu, one_more_conv=self.config.one_more_conv,attr_each_deconv=self.config.attr_each_deconv)
        self.D = Discriminator(self.config.image_size, self.config.max_conv_dim, len(self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)
        self.LD = Latent_Discriminator(self.config.image_size, self.config.max_conv_dim, len(self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.g_layers, self.config.shortcut_layers)
        print(self.G)
        print(self.D)
        print(self.LD)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size)

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

    def save_checkpoint(self):
        def save_one_model(model,optimizer,name):
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,self.current_iteration)))
        save_one_model(self.G,self.optimizer_G,'G')
        save_one_model(self.D,self.optimizer_D,'D')
        save_one_model(self.LD,self.optimizer_LD,'LD')

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            self.G.to(self.device)
            self.D.to(self.device)
            self.LD.to(self.device)
            return
        def load_one_model(model,optimizer,name):
            G_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,self.config.checkpoint)),map_location=torch.device('cpu'))
            G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
            model.load_state_dict(G_to_load)
            model.to(self.device)
            if self.config.mode == 'train':
                optimizer.load_state_dict(G_checkpoint['optimizer'])
        load_one_model(self.G,self.optimizer_G if self.config.mode=='train' else None,'G')
        load_one_model(self.D,self.optimizer_D if self.config.mode=='train' else None,'D')
        load_one_model(self.LD,self.optimizer_LD if self.config.mode=='train' else None,'LD')

        self.current_iteration = self.config.checkpoint

    def denorm(self, x):
        #get from [-1,1] to [0,1]
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing: linearly sample attribute"""

        c_trg_list = []
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-5.0, 5.0, 10)
            #alphas = [torch.FloatTensor([alpha]) for alpha in alphas]
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss."""
        #TODO change to l2 loss (or l1)
        return F.l1_loss(logit,target)/ logit.size(0)
        #return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

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


    def compute_sample_grid(self,x_sample,c_sample_list,c_org_sample,path,writer=False):
        x_sample = x_sample.to(self.device)
        x_fake_list = [x_sample]
        for c_trg_sample in c_sample_list:
            attr_diff = c_trg_sample.to(self.device) - c_org_sample.to(self.device)
            attr_diff = attr_diff if self.config.use_attr_diff else c_trg_sample #* self.config.thres_int
            fake_image,_=self.G(x_sample, attr_diff.to(self.device))
            out_disc, out_att = self.D(fake_image.detach())

            #write target and predicted score
            for im in range(fake_image.shape[0]):
                #image=(fake_image[im].cpu().detach().numpy().transpose((1,2,0))*127.5+127.5).astype(np.uint8)
                image=np.zeros((128,128,3), np.uint8)
                for i in range(attr_diff.shape[1]):
                    cv2.putText(image, "%.2f"%(c_trg_sample[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
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
    def run(self):
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'train':
                self.train()
            else:
                self.test_pca()
                #self.test()
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
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.optimizer_LD = optim.Adam(self.LD.parameters(), self.config.ld_lr, [self.config.beta1, self.config.beta2])
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_LD = optim.lr_scheduler.StepLR(self.optimizer_LD, step_size=self.config.lr_decay_iters, gamma=0.1)

        self.load_checkpoint()
        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(self.G, device_ids=list(range(self.config.ngpu)))
            self.D = nn.DataParallel(self.D, device_ids=list(range(self.config.ngpu)))
            self.LD = nn.DataParallel(self.LD, device_ids=list(range(self.config.ngpu)))

        # samples used for testing (linear samples) the net
        val_iter = iter(self.data_loader.val_loader)
        Ia_sample, a_sample = next(val_iter)
        Ia_sample = Ia_sample.to(self.device)
        b_samples = self.create_labels(a_sample, self.config.attrs)
        b_samples.insert(0, a_sample)  # reconstruction

        self.g_lr = self.lr_scheduler_G.get_lr()[0]
        self.d_lr = self.lr_scheduler_D.get_lr()[0]
        self.ld_lr = self.lr_scheduler_LD.get_lr()[0]
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
                    Ia, a_att = next(data_iter)
                except:
                    data_iter = iter(self.data_loader.train_loader)
                    Ia, a_att = next(data_iter)

                # generate target domain labels randomly
                b_att = a_att + torch.randn_like(a_att)*self.config.gaussian_stddev

                a_att_copy = a_att.clone()
                b_att_copy = b_att.clone()

                Ia = Ia.to(self.device)         # input images
                a_att_copy = a_att_copy.to(self.device)           # original domain labels
                b_att_copy = b_att_copy.to(self.device)           # target domain labels
                a_att = a_att.to(self.device)   # labels for computing classification loss
                b_att = b_att.to(self.device)   # labels for computing classification loss

                attr_diff = b_att_copy - a_att_copy
                attr_diff = attr_diff if self.config.use_attr_diff else b_att_copy
                scalars = {}
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                if self.config.use_image_disc or self.config.use_classifier_generator:
                    self.G.eval()
                    self.D.train()
                    self.LD.eval()

                    # input is the real image Ia
                    out_disc_real, out_att_real = self.D(Ia)

                    d_loss = 0
                    if self.config.use_image_disc:
                        # fake image Ib_hat
                        Ib_hat,_ = self.G(Ia, attr_diff)
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
                        d_loss_att = self.classification_loss(out_att_real, a_att)
                        d_loss += self.config.lambda_d_att * d_loss_att
                        scalars['D/loss_att'] = d_loss_att.item()


                    # backward and optimize
                    self.optimizer_D.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.optimizer_D.step()
                    # summarize
                    scalars['D/loss'] = d_loss.item()


                # =================================================================================== #
                #                         3. Train the latent discriminator                           #
                # =================================================================================== #
                if self.config.use_latent_disc:
                    self.G.eval()
                    self.D.eval()
                    self.LD.train()
                    # compute disc loss on encoded image
                    _,z = self.G(Ia, a_att_copy - a_att_copy if self.config.use_attr_diff else a_att_copy)
                    out_att = self.LD(z)

                    #classification loss
                    ld_loss = self.classification_loss(out_att, a_att)*self.config.lambda_ld

                    # backward and optimize
                    self.optimizer_LD.zero_grad()
                    ld_loss.backward(retain_graph=True)
                    self.optimizer_LD.step()

                    # summarize
                    scalars = {}
                    scalars['LD/loss'] = ld_loss.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                if (self.current_iteration + 1) % self.config.n_critic == 0: # and i>20000:  
                    self.G.train()
                    self.D.eval()
                    self.LD.eval()

                    # target-to-original domain : Ia_hat -> reconstruction
                    Ia_hat,z = self.G(Ia, a_att_copy - a_att_copy if self.config.use_attr_diff else a_att_copy)
                    if self.config.use_l1_rec_loss:
                        g_loss_rec = torch.mean(torch.abs(Ia - Ia_hat))
                    else:
                        g_loss_rec = ((Ia - Ia_hat) ** 2).mean()
                    g_loss = self.config.lambda_g_rec * g_loss_rec

                    if self.config.use_latent_disc:
                        out_att = self.LD(z)
                        g_loss_latent = -self.classification_loss(out_att, a_att)
                        g_loss += self.config.lambda_g_latent * g_loss_latent
                        scalars['G/loss_latent'] = g_loss_latent.item()

                    if self.config.use_image_disc or self.config.use_classifier_generator:
                        # original-to-target domain : Ib_hat -> GAN + classif
                        Ib_hat,_ = self.G(Ia, attr_diff)
                        out_disc, out_att = self.D(Ib_hat)
                        if self.config.use_image_disc:                    
                            g_loss_adv = - torch.mean(out_disc)
                            g_loss += self.config.lambda_adv * g_loss_adv
                            scalars['G/loss_adv'] = g_loss_adv.item()
                        if self.config.use_classifier_generator:
                            g_loss_att = self.classification_loss(out_att, b_att)
                            g_loss += self.config.lambda_g_att * g_loss_att
                            scalars['G/loss_att'] = g_loss_att.item()


                    # backward and optimize
                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    self.optimizer_G.step()

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



    def test_classif(self):
        self.load_checkpoint()
        self.D.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        self.D.eval()
        all_figs=[]; ax=[]
        with torch.no_grad():
            for i in range(len(self.config.attrs)):
                all_figs.append([plt.figure(figsize=(8,12)),plt.figure(figsize=(8,12))])
                ax.append([all_figs[i][0].gca(),all_figs[i][1].gca()])
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                x_real = x_real.to(self.device)
                out_disc, out_att = self.D(x_real)
                for j in range(x_real.shape[0]):
                    for att in range(len(self.config.attrs)):
                        _imscatter(c_org[j][att].cpu(), np.random.random(),
                                image=x_real[j].cpu(),
                                color='white',
                                zoom=0.1,
                                ax=ax[att][0])
                        
                        _imscatter(out_att[j][att].cpu(), np.random.random(),
                                image=x_real[j].cpu(),
                                color='white',
                                zoom=0.1,
                                ax=ax[att][1])
                        #print(out_att[j][att])
            for i in range(len(self.config.attrs)):
                result_path = os.path.join(self.config.result_dir, 'scores_{}_{}.jpg'.format(i,"0"))
                print(result_path)
                all_figs[i][0].savefig(result_path, dpi=300, bbox_inches='tight')
                #all_figs[i][0].show()
                result_path = os.path.join(self.config.result_dir, 'scores_{}_{}.jpg'.format(i,1))
                all_figs[i][1].savefig(result_path, dpi=300, bbox_inches='tight')

    def test(self):
        self.load_checkpoint()
        self.G.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                c_trg_list = self.create_labels(c_org, self.config.attrs)
                c_trg_list.insert(0, c_org)
                self.compute_sample_grid(x_real,c_trg_list,c_org,os.path.join(self.config.result_dir, 'sample_{}.jpg'.format(i + 1)),writer=False)

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
                partial_embs.append(encoded.detach().cpu())
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
                numpy_encoded=encoded.detach().cpu().numpy()
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
                    save_image(self.denorm(x_concat.data.cpu()),os.path.join(self.config.result_dir, 'sample_ica_axis_{}_{}.jpg'.format(axis,i + 1)),nrow=1, padding=0)




    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
