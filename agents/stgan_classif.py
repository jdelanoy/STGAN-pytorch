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
        D_state  = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
        }
        D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
        torch.save(D_state, os.path.join(self.config.checkpoint_dir, D_filename))

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            self.D.to(self.device)
            return
        D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
        D_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, D_filename),map_location=torch.device('cpu'))
        D_to_load = {k.replace('module.', ''): v for k, v in D_checkpoint['state_dict'].items()}
        self.current_iteration = self.config.checkpoint
        self.D.load_state_dict(D_to_load)
        self.D.to(self.device)
        if self.config.mode == 'train':
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
            c_trg = c_org.clone()
            c_trg[:, i] = c_trg[:, i] + torch.randn_like(c_trg[:, i])*self.config.gaussian_stddev
            # if i in hair_color_indices:  # set one hair color to 1 and the rest to 0
            #     c_trg[:, i] = 1
            #     for j in hair_color_indices:
            #         if j != i:
            #             c_trg[:, j] = 0
            # else:
            #     c_trg[:, i] = (c_trg[:, i] == 0)  # reverse attribute value

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

    def run(self):
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'train':
                self.train()
            else:
                self.test_classif()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
        finally:
            self.finalize()


    def train(self):
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)

        self.load_checkpoint()
        if self.cuda and self.config.ngpu > 1:
            self.D = nn.DataParallel(self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)
        x_sample, c_org_sample = next(val_iter)
        x_sample = x_sample.to(self.device)
        c_sample_list = self.create_labels(c_org_sample, self.config.attrs)
        c_sample_list.insert(0, c_org_sample)  # reconstruction

        self.d_lr = self.lr_scheduler_D.get_lr()[0]

        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        for i in range(self.current_iteration, self.config.max_iters):
            #print(self.config.max_iters)
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


            c_org = label_org.clone()

            x_real = x_real.to(self.device)         # input images
            c_org = c_org.to(self.device)           # original domain labels
            label_org = label_org.to(self.device)   # labels for computing classification loss

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # compute loss with real images
            out_src, out_cls = self.D(x_real)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # backward and optimize
            d_loss = d_loss_cls
            self.optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = d_loss.item()
            scalars['D/loss_cls'] = d_loss_cls.item()

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


            if self.current_iteration % self.config.checkpoint_step == 0:
                self.save_checkpoint()

            self.lr_scheduler_D.step()

    def test_classif(self):
        self.load_checkpoint()
        self.D.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.D.eval()
        all_figs=[]; ax=[]
        with torch.no_grad():
            for i in range(len(self.config.attrs)):
                all_figs.append([plt.figure(figsize=(8, 12)),plt.figure(figsize=(8, 12))])
                ax.append([all_figs[i][0].gca(),all_figs[i][1].gca()])
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                x_real = x_real.to(self.device)
                out_src, out_cls = self.D(x_real)
                for j in range(x_real.shape[0]):
                    for att in range(len(self.config.attrs)):
                        _imscatter(c_org[j][att], np.random.random(),
                                image=x_real[j],
                                color='white',
                                zoom=0.1,
                                ax=ax[att][0])
                        
                        _imscatter(out_cls[j][att], np.random.random(),
                                image=x_real[j],
                                color='white',
                                zoom=0.1,
                                ax=ax[att][1])
                        #print(out_cls[j][att])
            for i in range(len(self.config.attrs)):
                result_path = os.path.join(self.config.result_dir, 'scores_{}_{}.jpg'.format(i,0))
                print(result_path)
                all_figs[i][0].savefig(result_path, dpi=300, bbox_inches='tight')
                #all_figs[i][0].show()
                result_path = os.path.join(self.config.result_dir, 'scores_{}_{}.jpg'.format(i,1))
                all_figs[i][1].savefig(result_path, dpi=300, bbox_inches='tight')


    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
