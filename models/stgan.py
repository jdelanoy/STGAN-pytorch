import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np


class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, in_state_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(in_state_dim + n_attrs, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state

def get_encoder_layers(conv_dim=64, n_layers=5, max_dim = 1024, norm=nn.BatchNorm2d, bias=False, dropout=0, vgg_like=False):
    layers = []
    in_channels = 3
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer=[nn.Conv2d(in_channels, out_channels, 4, 2, 1,bias=bias)]
        if i > 0: #NOTE >= in AttGAN
            enc_layer.append(norm(out_channels, affine=True, track_running_stats=True))
        enc_layer.append(nn.LeakyReLU(0.2, inplace=True))
        if (vgg_like and i >= 3 and i<n_layers-1):
            enc_layer += [nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),norm(out_channels, affine=True, track_running_stats=True),nn.LeakyReLU(0.2, inplace=True)]
            enc_layer += [nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),norm(out_channels, affine=True, track_running_stats=True),nn.LeakyReLU(0.2, inplace=True)]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers

class Generator(nn.Module):
    def __init__(self, attr_dim, conv_dim=64, n_layers=5, max_dim=1024, shortcut_layers=2, stu_kernel_size=3, use_stu=True, one_more_conv=True, n_attr_deconv=1, vgg_like=False):
        super(Generator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu
        self.n_attr_deconv = n_attr_deconv

        ##### build encoder
        enc_layers=get_encoder_layers(conv_dim,n_layers,max_dim,bias=True,vgg_like=vgg_like) #NOTE bias=false for STGAN
        self.encoder = nn.ModuleList(enc_layers)

        self.stu = nn.ModuleList()
        ##### build decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.n_layers)):
            #size if inputs/outputs
            dec_out = min(max_dim,conv_dim * 2 ** (i-1)) #NOTE ou i in STGAN
            dec_in = min(max_dim,conv_dim * 2 ** (i)) #NOTE ou i+1 in STGAN
            enc_size = min(max_dim,conv_dim * 2 ** (i))

            #if i == self.n_layers-1 or attr_each_deconv: dec_in = dec_in + attr_dim #concatenate attribute
            if i >= self.n_layers - self.n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
            if i >= self.n_layers - 1 - self.shortcut_layers and i != self.n_layers-1: # skip connection
                dec_in = dec_in + enc_size
                if use_stu:
                    self.stu.append(ConvGRUCell(self.n_attrs, enc_size, enc_size, min(max_dim,enc_size*2), stu_kernel_size))
            #print(i,dec_out,dec_in,enc_size)

            if i > 0:
                if vgg_like and i > 3:
                    self.decoder.append(nn.Sequential(
                        #nn.ConvTranspose2d(dec_in, dec_out, 4, 2, 1, bias=False),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dec_out, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        #nn.ConvTranspose2d(dec_in, dec_out, 4, 2, 1, bias=False),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True)
                    ))
            else: #last layer
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, conv_dim // 4, 3, 1, 1),
                        #nn.ConvTranspose2d(dec_in, conv_dim // 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(conv_dim // 4),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, 3, 3, 1, 1),
                        #nn.ConvTranspose2d(dec_in, 3, 4, 2, 1, bias=False),
                        nn.Tanh()
                    ))

    def decode(self, z, a):
        #first decoder step
        out = z
        a = a.view((out.size(0), self.n_attrs, 1, 1))
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            out = dec_layer(out)
        return out

    def forward(self, x, a):
        # propagate encoder layers
        encoded = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            encoded.append(x_)

        #first decoder step
        out = encoded[-1]
        stu_state = encoded[-1]
        a = a.view((out.size(0), self.n_attrs, 1, 1))

        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.shortcut_layers:
                #do shortcut connection
                if self.use_stu:
                    stu_out, stu_state = self.stu[i-1](encoded[-(i+1)], stu_state, a)
                    out = torch.cat([out, stu_out], dim=1)
                else:
                    out = torch.cat([out, encoded[-(i+1)]], dim=1)
            out = dec_layer(out)
        return out,encoded


class DisentangledGenerator(nn.Module):
    def __init__(self, attr_dim, n_embeddings, conv_dim=64, n_layers=5, max_dim=1024, shortcut_layers=2, stu_kernel_size=3, use_stu=True, one_more_conv=True, n_attr_deconv=1, vgg_like=False):
        super(DisentangledGenerator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu
        self.n_attr_deconv = n_attr_deconv

        ##### build encoder
        enc_layers=get_encoder_layers(conv_dim>>1,n_layers,max_dim>>1,bias=True,vgg_like=vgg_like) #NOTE bias=false for STGAN
        self.encoder = nn.ModuleList(enc_layers)

        self.stu = nn.ModuleList()
        ##### build decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.n_layers)):
            #size if inputs/outputs
            dec_out = min(max_dim,conv_dim * 2 ** (i-1)) #NOTE ou i in STGAN
            dec_in = min(max_dim,conv_dim * 2 ** (i)) #NOTE ou i+1 in STGAN
            enc_size = min(max_dim,conv_dim * 2 ** (i))>>1

            #if i == self.n_layers-1 or attr_each_deconv: dec_in = dec_in + attr_dim #concatenate attribute
            if i >= self.n_layers - self.n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
            if i == self.n_layers-1: dec_in = enc_size
            if i >= self.n_layers - 1 - self.shortcut_layers: # and i != self.n_layers-1: # skip connection 
                dec_in = dec_in + n_embeddings * (enc_size)
                if use_stu:
                    self.stu.append(ConvGRUCell(self.n_attrs, enc_size, enc_size, min(max_dim,enc_size*2), stu_kernel_size))
            print(i,dec_in,dec_out,enc_size)

            if i > 0:
                if vgg_like and i > 3:
                    self.decoder.append(nn.Sequential(
                        #nn.ConvTranspose2d(dec_in, dec_out, 4, 2, 1, bias=False),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dec_out, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        #nn.ConvTranspose2d(dec_in, dec_out, 4, 2, 1, bias=False),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, dec_out, 3, 1, 1),
                        nn.BatchNorm2d(dec_out),
                        nn.ReLU(inplace=True)
                    ))
            else: #last layer
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, conv_dim // 4, 3, 1, 1),
                        #nn.ConvTranspose2d(dec_in, conv_dim // 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(conv_dim // 4),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, 3, 3, 1, 1),
                        #nn.ConvTranspose2d(dec_in, 3, 4, 2, 1, bias=False),
                        nn.Tanh()
                    ))

    def decode(self, z, a):
        #first decoder step
        out = z
        a = a.view((out.size(0), self.n_attrs, 1, 1))
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            out = dec_layer(out)
        return out

    def forward(self, x, a, encodings):
        # propagate encoder layers
        encoded = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            encoded.append(x_)

        #first decoder step
        out = encoded[-1]
        stu_state = encoded[-1]
        a = a.view((out.size(0), self.n_attrs, 1, 1))

        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if  i <= self.shortcut_layers:
                #do shortcut connection
                for enc in encodings:
                    out = torch.cat([out, enc[self.n_layers-1-i]], dim=1)
            out = dec_layer(out)
        return out,encoded


class Latent_Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, conv_dim=64, fc_dim=1024, n_layers=5, shortcut_layers=2,vgg_like=False,tanh=True):
        super(Latent_Discriminator, self).__init__()
        layers = []
        n_dis_layers = int(np.log2(image_size))
        layers=get_encoder_layers(conv_dim,n_dis_layers, max_dim, norm=nn.BatchNorm2d,bias=True,dropout=0.3,vgg_like=vgg_like)
        self.conv = nn.Sequential(*layers[n_layers-shortcut_layers:])

        out_conv = min(max_dim,conv_dim * 2 ** (n_dis_layers - 1))
        if (tanh):
            self.fc_att = nn.Sequential(
                nn.Linear(out_conv, fc_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(fc_dim, attr_dim),
                nn.Tanh()
            )
        else:
            self.fc_att = nn.Sequential(
                nn.Linear(out_conv, fc_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(fc_dim, attr_dim)
            )
    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att


class Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, conv_dim=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        layers=get_encoder_layers(conv_dim,n_layers, max_dim, norm=nn.InstanceNorm2d,bias=True)
        self.conv = nn.Sequential(*layers)

        feature_size = image_size // 2**n_layers
        out_conv = min(max_dim,conv_dim * 2 ** (n_layers - 1))
        self.fc_adv = nn.Sequential(
            nn.Linear(out_conv * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(out_conv * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att


class Classifier(nn.Module):
    def __init__(self, image_size=128, max_dim=512, n_classes=10, conv_dim=64, fc_dim=1024, n_layers=5, vgg_like=False):
        super(Classifier, self).__init__()
        layers = []
        layers=get_encoder_layers(conv_dim,n_layers, max_dim, norm=nn.InstanceNorm2d,bias=True, vgg_like=vgg_like)
        self.conv = nn.Sequential(*layers)

        feature_size = image_size // 2**n_layers
        out_conv = min(max_dim,conv_dim * 2 ** (n_layers - 1))

        self.fc_att = nn.Sequential(
            nn.Linear(out_conv * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, n_classes)
            #nn.Tanh()
        )

    def forward(self, x):
        encoded = []
        y = x
        for layer in self.conv:
            y = layer(y)
            encoded.append(y)
        #y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return encoded,logit_att



if __name__ == '__main__':
    gen = Generator(5, conv_dim=32, n_layers=6, shortcut_layers=0, max_dim=512, use_stu=False, one_more_conv=False)

    print(gen)
    #summary(gen, [(3, 128, 128), (5,)], device='cpu')

    dis = Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=7)
    print(dis)
    summary(dis, (3, 128, 128), device='cpu')

    dis = Latent_Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=6,shortcut_layers=2)
    print(dis)
