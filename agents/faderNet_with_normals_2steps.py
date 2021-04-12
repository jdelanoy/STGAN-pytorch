import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGeneratorWithNormals2Steps,FaderNetGeneratorWithNormals, Discriminator, Latent_Discriminator, Unet
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.faderNet import FaderNet



class FaderNetWithNormals2Steps(FaderNet):
    def __init__(self, config):
        super(FaderNetWithNormals2Steps, self).__init__(config)

        ###only change generator
        self.G = FaderNetGeneratorWithNormals2Steps(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals, normalization=self.norm, first_conv=config.first_conv, n_bottlenecks=config.n_bottlenecks)
        print(self.G)

        ### load the normal predictor network
        self.normal_G = Unet(conv_dim=config.g_conv_dim_normals,n_layers=config.g_layers_normals,max_dim=config.max_conv_dim_normals, im_channels=config.img_channels, skip_connections=config.skip_connections_normals, vgg_like=config.vgg_like_normals)
        #self.load_model_from_path(self.normal_G,config.normal_predictor_checkpoint)
        self.normal_G.eval()

        #load the small FaderNet
        self.G_small = FaderNetGeneratorWithNormals(conv_dim=32,n_layers=6,max_dim=512, im_channels=config.img_channels, skip_connections=0, vgg_like=0, attr_dim=len(config.attrs), n_attr_deconv=1, n_concat_normals=4, normalization=self.norm, first_conv=False, n_bottlenecks=2)
        self.load_model_from_path(self.G_small,config.faderNet_checkpoint)


        self.logger.info("FaderNet with normals in 2 steps ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,encodings,att):
        normals= self.get_normals()
        fn_output, fn_features = self.get_fadernet_output(att)
        return self.G.decode(att,bneck,normals,fn_output,encodings)

    def init_sample_grid(self):
        x_fake_list = [self.get_fadernet_output(self.batch_a_att)[0],self.batch_Ia[:,:3]]
        return x_fake_list



    def get_normals(self):
        return self.batch_normals[:,:3]
        normals=self.normal_G(self.batch_Ia)
        return normals*self.batch_Ia[:,3:]


    def get_fadernet_output(self,att):
        encodings,z,_ = self.G_small.encode(self.batch_Ia)
        return self.G_small.decode_with_features(att,z,self.get_normals(),encodings)




