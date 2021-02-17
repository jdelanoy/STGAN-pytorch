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
from agents.faderNet import FaderNet



class FaderNetWithNormals(FaderNet):
    def __init__(self, config):
        super(FaderNetWithNormals, self).__init__(config)

        ###only change generator
        self.G = FaderNetGeneratorWithNormals(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals)
        print(self.G)

        ### load the normal predictor network
        self.normal_G = Unet(conv_dim=config.g_conv_dim_normals,n_layers=config.g_layers_normals,max_dim=config.max_conv_dim_normals, im_channels=config.img_channels, skip_connections=config.skip_connections_normals, vgg_like=config.vgg_like_normals)
        self.load_model_from_path(self.normal_G,config.normal_predictor_checkpoint)
        self.normal_G.eval()

        self.logger.info("FaderNet with normals ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,batch,bneck,encodings,new_attr=None):
        x_sample, normals, illum, att = batch
        if new_attr != None: att=new_attr
        normals=torch.cat([normals[:,:3],illum],dim=1) #self.get_normals(x_sample)
        return self.G.decode(att,bneck,normals,encodings)

    def init_sample_grid(self,batch):
        x_sample, normals, illum, att = batch
        x_fake_list = [normals[:,:3],torch.cat([illum,illum,illum],dim=1),x_sample[:,:3]]
        return x_fake_list



    def get_normals(self, img):
        normals=self.normal_G(img)
        return normals*img[:,3:]







