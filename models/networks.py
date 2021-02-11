import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

from models.blocks import * 


def build_encoder_layers(conv_dim=64, n_layers=6, max_dim = 512, im_channels = 3, activation='relu', normalization='batch',dropout=0, vgg_like=0):
    bias = normalization == 'none'  # use bias only if we do not use a normalization layer  #TODO old archi
    kernel_sizes=[4,4,4,4,4,4,4] #[7,5,5,3,3,3,3]  #TODO old archi
    
    layers = []
    in_channels = im_channels
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer=[ConvReluBn(nn.Conv2d(in_channels, out_channels, kernel_sizes[i], 2, (kernel_sizes[i]-1)//2,bias=bias),activation,normalization=normalization)] #TODO
        if (i >= n_layers-1-vgg_like and i<n_layers-1):
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),activation,normalization)]
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),activation,normalization)]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers



class Encoder(nn.Module):
    def __init__(self, conv_dim, n_layers, max_dim, im_channels, vgg_like):
        super(Encoder, self).__init__()
        act='leaky_relu'
        norm='batch'
        bias = norm == 'none'  # use bias only if we do not use a normalization layer 
        enc_layers=build_encoder_layers(conv_dim,n_layers,max_dim, im_channels,normalization=norm,activation=act,vgg_like=vgg_like) 
        self.encoder = nn.ModuleList(enc_layers)
        b_dim=min(max_dim,conv_dim * 2 ** (n_layers-1))
        self.bottleneck = nn.ModuleList([  #TODO old archi
            BottleneckBlock(b_dim, b_dim, act, norm, bias=bias),
            BottleneckBlock(b_dim, b_dim, act, norm, bias=bias),
        ])
    #return [encodings,bneck]
    def encode(self,x):
        # Encoder
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            x_encoder.append(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        return x_encoder, x



def build_decoder_layers(conv_dim=64, n_layers=6, max_dim=512, im_channels=3, skip_connections=0,attr_dim=0,n_attr_deconv=0, vgg_like=0, n_branches=1,activation='relu', normalization='batch', add_normal_map=0):
    bias=normalization=='none'  #TODO old archi
    decoder = nn.ModuleList()
    for i in reversed(range(n_layers)):
        #size of inputs/outputs
        dec_out = min(max_dim,conv_dim * 2 ** (i-1))
        dec_in = min(max_dim,conv_dim * 2 ** (i))
        enc_size = min(max_dim,conv_dim * 2 ** (i)) #corresponding encoding size (for skip connections)
        
        if i == n_layers-1: dec_in = enc_size * n_branches
        if i >= n_layers - n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
        if i >= n_layers - 1 - skip_connections and i != n_layers-1: # skip connection: n_branches-1 or 1 feature map
            dec_in = dec_in + max(1,n_branches-1)*enc_size 
        if (i==0): dec_out=conv_dim // 4 
        if (i < add_normal_map): dec_in += 3

        dec_layer=[ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias),activation=activation,normalization=normalization)] #TODO
        if (vgg_like > 0 and i >= n_layers - vgg_like) or (i==0 and add_normal_map):
            dec_layer+=[ConvReluBn(nn.Conv2d(dec_out, dec_out, 3, 1, 1,bias=bias),activation,normalization)]
        decoder.append(nn.Sequential(*dec_layer))

    last_conv = nn.ConvTranspose2d(conv_dim // 4, im_channels, 3, 1, 1, bias=True)
    return decoder, last_conv


class Unet(nn.Module):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2,vgg_like=0):
        super(Unet, self).__init__()
        self.n_layers = n_layers
        self.skip_connections = min(skip_connections, n_layers - 1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) #TODO old archi

        ##### build encoder
        self.encoder = Encoder(conv_dim,n_layers,max_dim,im_channels,vgg_like)
        ##### build decoder
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, skip_connections=skip_connections,vgg_like=vgg_like)

    #return [encodings,bneck]
    def encode(self,x):
        return self.encoder.encode(x)

    def decode(self, bneck, encodings):
        #expand dimensions of a
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if 0 < i <= self.skip_connections:
                out = torch.cat([out, encodings[-(i+1)]], dim=1)
            out = dec_layer(self.up(out))
        x = self.last_conv(out) #TODO old archi
        x = torch.tanh(x)
        x = x / torch.sqrt((x**2).sum(dim=1,keepdims=True))
        return x

    def forward(self, x):
        # propagate encoder layers
        encodings,z = self.encode(x)
        return self.decode(z,encodings)


class FaderNetGenerator(Unet):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2,vgg_like=0,attr_dim=1,n_attr_deconv=1):
        super(FaderNetGenerator, self).__init__(conv_dim, n_layers, max_dim, im_channels, skip_connections,vgg_like)
        self.attr_dim = attr_dim
        self.n_attr_deconv = n_attr_deconv
        ##### change decoder : get attribute as input
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, skip_connections=skip_connections,vgg_like=vgg_like, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv)

    def decode(self, a, bneck, encodings):
        #expand dimensions of a
        a = a.view((bneck.size(0), self.attr_dim, 1, 1))
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.attr_dim, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.skip_connections:
                #do shortcut connection, not taking the first encoding (mat)
                out = torch.cat([out, encodings[-(i+1)]], dim=1)
            out = dec_layer(self.up(out))
        x = self.last_conv(out) #TODO old archi
        x = torch.tanh(x)

        return x

    def forward(self, x,a):
        # propagate encoder layers
        encodings,z = self.encode(x)
        return self.decode(a,z,encodings)


class FaderNetGeneratorWithNormals(Unet):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2,vgg_like=0,attr_dim=1,n_attr_deconv=1,n_concat_normals=1):
        super(FaderNetGeneratorWithNormals, self).__init__(conv_dim, n_layers, max_dim, im_channels, skip_connections,vgg_like)
        self.attr_dim = attr_dim
        self.n_attr_deconv = n_attr_deconv
        self.n_concat_normals = n_concat_normals
        ##### change decoder : get attribute as input
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, skip_connections=skip_connections,vgg_like=vgg_like, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv, add_normal_map=self.n_concat_normals)

    def decode(self, a, bneck, normals, encodings):
        #build pyramid of normals (should be directly provided in future version)
        normal_pyramid=[normals]
        for i in range(self.n_concat_normals-1):
            normal_pyramid.insert(0,nn.functional.interpolate(normal_pyramid[0], mode='bilinear', align_corners=True, scale_factor=0.5))
        #expand dimensions of a
        a = a.view((bneck.size(0), self.attr_dim, 1, 1))
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.attr_dim, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.skip_connections:
                #do shortcut connection, not taking the first encoding (mat)
                out = torch.cat([out, encodings[-(i+1)]], dim=1)
            if i >= self.n_layers-self.n_concat_normals:
                #add normal map to the last deconv
                out = dec_layer(torch.cat([self.up(out), normal_pyramid[i-(self.n_layers-self.n_concat_normals)]], dim=1))
            else:
                out = dec_layer(self.up(out))
        x = self.last_conv(out) #TODO old archi
        x = torch.tanh(x)

        return x

    def forward(self, x,a,normals):
        # propagate encoder layers
        encodings,z = self.encode(x)
        return self.decode(a,z,normals,encodings)



def FC_layers(in_dim,fc_dim,out_dim,tanh):
    layers=[nn.Linear(in_dim, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, out_dim)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Latent_Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels = 3,conv_dim=64, fc_dim=1024, n_layers=5, skip_connections=2,vgg_like=0):
        super(Latent_Discriminator, self).__init__()
        layers = []
        n_dis_layers = int(np.log2(image_size))
        layers=build_encoder_layers(conv_dim,n_dis_layers, max_dim, im_channels, normalization='batch',activation='leaky_relu',dropout=0.3,vgg_like=vgg_like) #TODO act
        self.conv = nn.Sequential(*layers[n_layers-skip_connections:])

        out_conv = min(max_dim,conv_dim * 2 ** (n_dis_layers - 1))
        self.fc_att = FC_layers(out_conv,fc_dim,attr_dim,True)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att


class Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels = 3, conv_dim=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        layers=build_encoder_layers(conv_dim,n_layers, max_dim, im_channels, normalization='batch')
        self.conv = nn.Sequential(*layers)

        feature_size = image_size // 2**n_layers
        out_conv = min(max_dim,conv_dim * 2 ** (n_layers - 1))
        self.fc_adv = FC_layers(out_conv * feature_size ** 2,fc_dim,1,False)
        #self.fc_att = FC_layers(out_conv * feature_size ** 2,fc_dim,attr_dim,True)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        #logit_att = self.fc_att(y)
        return logit_adv




if __name__ == '__main__':
    gen = Generator(5, conv_dim=32, n_layers=6, skip_connections=0, max_dim=512,vgg_like=True)

    print(gen)
    #summary(gen, [(3, 128, 128), (5,)], device='cpu')

    # dis = Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=7)
    # print(dis)
    # #summary(dis, (3, 128, 128), device='cpu')

    # dis = Latent_Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=6,skip_connections=2)
    # print(dis)
