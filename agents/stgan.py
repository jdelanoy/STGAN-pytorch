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

from datasets import *
from models.stgan import Generator, Discriminator
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

        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers, self.config.shortcut_layers, use_stu=self.config.use_stu, one_more_conv=self.config.one_more_conv)
        self.D = Discriminator(self.config.image_size, len(self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)

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
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
        }
        D_state  = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
        }
        G_filename = 'G_{}.pth.tar'.format(self.current_iteration)
        D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
        torch.save(G_state, os.path.join(self.config.checkpoint_dir, G_filename))
        torch.save(D_state, os.path.join(self.config.checkpoint_dir, D_filename))

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            self.G.to(self.device)
            self.D.to(self.device)
            return
        G_filename = 'G_{}.pth.tar'.format(self.config.checkpoint)
        D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
        G_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, G_filename),map_location=torch.device('cpu'))
        D_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, D_filename),map_location=torch.device('cpu'))
        G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
        D_to_load = {k.replace('module.', ''): v for k, v in D_checkpoint['state_dict'].items()}
        self.current_iteration = self.config.checkpoint
        self.G.load_state_dict(G_to_load)
        self.D.load_state_dict(D_to_load)
        self.G.to(self.device)
        self.D.to(self.device)
        if self.config.mode == 'train':
            self.optimizer_G.load_state_dict(G_checkpoint['optimizer'])
            self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])

    def denorm(self, x):
        #get from [-1,1] to [0,1]
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        #TODO add gaussian noise 
        # # get hair color indices
        # hair_color_indices = []
        # for i, attr_name in enumerate(selected_attrs):
        #     if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
        #         hair_color_indices.append(i)

        c_trg_list = []
        for i in range(len(selected_attrs)):
            alphas = np.linspace(-5.0, 5.0, 10)
            #alphas = [torch.FloatTensor([alpha]) for alpha in alphas]
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) #c_trg[:, i] + torch.randn_like(c_trg[:, i])*self.config.gaussian_stddev   A
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
            attr_diff = attr_diff #* self.config.thres_int
            fake_image=self.G(x_sample, attr_diff.to(self.device))
            out_src, out_cls = self.D(fake_image.detach())
            #print(fake_image.shape) 
            #print(attr_diff.shape)
            for im in range(fake_image.shape[0]):
                #image=(fake_image[im].cpu().detach().numpy().transpose((1,2,0))*127.5+127.5).astype(np.uint8)
                image=np.zeros((128,128,3), np.uint8)
                for i in range(attr_diff.shape[1]):
                    cv2.putText(image, "%.2f"%(c_trg_sample[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
                    cv2.putText(image, "%.2f"%(out_cls[im][i].item()), (10,14*(i+7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
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
                self.test()
                self.test_classif()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
        finally:
            self.finalize()


    def train(self):
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)

        self.load_checkpoint()
        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(self.G, device_ids=list(range(self.config.ngpu)))
            self.D = nn.DataParallel(self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)
        x_sample, c_org_sample = next(val_iter)
        x_sample = x_sample.to(self.device)
        c_sample_list = self.create_labels(c_org_sample, self.config.attrs)
        c_sample_list.insert(0, c_org_sample)  # reconstruction

        self.g_lr = self.lr_scheduler_G.get_lr()[0]
        self.d_lr = self.lr_scheduler_D.get_lr()[0]
        print(self.g_lr,self.d_lr)
        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        for i in range(self.current_iteration, self.config.max_iters):
            #print(self.config.max_iters)
            self.G.train()
            self.D.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # fetch real images and labels
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader.train_loader)
                x_real, label_org = next(data_iter)

            # generate target domain labels randomly
            #TODO add gaussian noise
            #rand_idx = torch.randperm(label_org.size(0))
            #label_trg = label_org[rand_idx]
            #print(label_org)
            label_trg = label_org + torch.randn_like(label_org)*self.config.gaussian_stddev
            #print(torch.randn_like(label_org))#*self.config.gaussian_stddev)
            #print(label_trg)

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)         # input images
            c_org = c_org.to(self.device)           # original domain labels
            c_trg = c_trg.to(self.device)           # target domain labels
            label_org = label_org.to(self.device)   # labels for computing classification loss
            label_trg = label_trg.to(self.device)   # labels for computing classification loss

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # compute loss with real images
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # compute loss with fake images
            attr_diff = c_trg - c_org
            attr_diff = attr_diff #* torch.rand_like(attr_diff) * (2 * self.config.thres_int) #TODO why random?
            x_fake = self.G(x_real, attr_diff)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # compute loss for gradient penalty
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # backward and optimize
            d_loss_adv = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
            d_loss = d_loss_adv + self.config.lambda_att * d_loss_cls
            #d_loss=d_loss_cls
            self.optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = d_loss.item()
            scalars['D/loss_adv'] = d_loss_adv.item()
            scalars['D/loss_cls'] = d_loss_cls.item()
            scalars['D/loss_real'] = d_loss_real.item()
            scalars['D/loss_fake'] = d_loss_fake.item()
            scalars['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            if (i + 1) % self.config.n_critic == 0: # and i>20000:  
                # original-to-target domain
                x_fake = self.G(x_real, attr_diff)
                out_src, out_cls = self.D(x_fake)
                g_loss_adv = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # target-to-original domain
                x_reconst = self.G(x_real, c_org - c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # backward and optimize
                g_loss = g_loss_adv + self.config.lambda_g_rec * g_loss_rec + self.config.lambda_g_att * g_loss_cls
                #g_loss = self.config.lambda_g_rec * g_loss_rec + self.config.lambda_g_att * g_loss_cls
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # summarize
                scalars['G/loss'] = g_loss.item()
                scalars['G/loss_adv'] = g_loss_adv.item()
                scalars['G/loss_cls'] = g_loss_cls.item()
                scalars['G/loss_rec'] = g_loss_rec.item()

            self.current_iteration += 1

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            if self.current_iteration % self.config.summary_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print('Elapsed [{}], Iteration [{}/{}]'.format(et, self.current_iteration, self.config.max_iters))
                for tag, value in scalars.items():
                    self.writer.add_scalar(tag, value, self.current_iteration)

            if (self.current_iteration-1) % self.config.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    self.compute_sample_grid(x_sample,c_sample_list,c_org_sample,os.path.join(self.config.sample_dir, 'sample_{}.jpg'.format(self.current_iteration)),writer=True)


            if self.current_iteration % self.config.checkpoint_step == 0:
                self.save_checkpoint()

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
            scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]
            #print(self.g_lr,self.d_lr)

    def test_classif(self):
        self.load_checkpoint()
        self.G.to(self.device)
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
                out_src, out_cls = self.D(x_real)
                for j in range(x_real.shape[0]):
                    for att in range(len(self.config.attrs)):
                        _imscatter(c_org[j][att].cpu(), np.random.random(),
                                image=x_real[j].cpu(),
                                color='white',
                                zoom=0.1,
                                ax=ax[att][0])
                        
                        _imscatter(out_cls[j][att].cpu(), np.random.random(),
                                image=x_real[j].cpu(),
                                color='white',
                                zoom=0.1,
                                ax=ax[att][1])
                        #print(out_cls[j][att])
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
                #x_real = x_real.to(self.device)
                # x_fake_list = [x_real]
                # for c_trg in c_trg_list:
                #     attr_diff = c_trg - c_org
                #     x_fake_list.append(self.G(x_real, attr_diff.to(self.device)))
                # x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.config.result_dir, 'sample_{}.jpg'.format(i + 1))
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
