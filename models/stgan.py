import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

from models.utils import activation_func, normalization_func






def masked_conv1x1(inp, out, stride=1, padding=0, bias=False):
    return nn.Conv2d(inp, out, 1, stride, padding, bias=bias)

def masked_conv3x3(inp, out, stride=1, padding=1, bias=False):
    return nn.Conv2d(inp, out, 3, stride, padding, bias=bias)

def masked_conv5x5(inp, out, stride=2, padding=2, bias=False):
    return nn.Conv2d(inp, out, 5, stride, padding, bias=bias)

def masked_conv7x7(inp, out, stride=2, padding=3, bias=False):
    return nn.Conv2d(inp, out, 7, stride, padding, bias=bias)

class UpAndConcat(nn.Module):
    def __init__(self):
        super(UpAndConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]

        # pad the image to allow for arbitrary inputs
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # concat both inputs
        x = torch.cat([x2, x1], dim=1)
        return x

class BottleneckBlock(nn.Module):

    def __init__(self, in_ch, out_ch, activation='relu', normalization='bn', bias=False):
        super(BottleneckBlock, self).__init__()

        self.activate = activation_func(activation)
        self.c1 = masked_conv3x3(in_ch, out_ch, bias=bias)
        self.bn1 = normalization_func(normalization)(out_ch)

        self.c2 = masked_conv3x3(out_ch, out_ch, bias=bias)
        self.bn2 = normalization_func(normalization)(out_ch)

    def forward(self, x):
        identity = x

        x = self.activate(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        x = self.activate(x + identity)
        return x




class ConvReluBn(nn.Module):
    def __init__(self, conv_layer, activation='relu', normalization='batch'):
        super(ConvReluBn, self).__init__()
        self.conv = conv_layer
        self.bn = normalization_func(normalization)(self.conv.out_channels)
        self.activate = activation_func(activation)

    def forward(self, x):
        x = self.activate(self.bn(self.conv(x)))
        return x


def get_encoder_layers(conv_dim=64, n_layers=5, max_dim = 1024, norm='batch',dropout=0, vgg_like=False):
    #TODO other option for kernel sizes: 7,5,5,3...
    bias = norm == 'none'  # use bias only if we do not use a norm layer
    kernel_sizes=[7,5,5,3,3,3,3]

    layers = []
    in_channels = 3
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer=[ConvReluBn(nn.Conv2d(in_channels, out_channels, kernel_sizes[i], 2, 1,bias=bias),'leaky_relu',norm)]
        if (vgg_like and i >= 3 and i<n_layers-1):
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),'leaky_relu',norm)]
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),'leaky_relu',norm)]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers



class Encoder(nn.Module):
    def __init__(self, conv_dim, n_layers, max_dim, vgg_like):
        super(Encoder, self).__init__()
        act='relu'
        norm='batch'
        bias = norm == 'none'
        enc_layers=get_encoder_layers(conv_dim,n_layers,max_dim,norm='batch',vgg_like=vgg_like) #NOTE bias=false for STGAN
        self.encoder = nn.ModuleList(enc_layers)

        self.bottleneck = nn.ModuleList([
            BottleneckBlock(max_dim, max_dim, act, norm, bias=bias),
            BottleneckBlock(max_dim, max_dim, act, norm, bias=bias),
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



def build_decoder_convs(attr_dim,conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like, n_branches):
    bias=False
    decoder = nn.ModuleList()
    for i in reversed(range(n_layers)):
        #size if inputs/outputs
        dec_out = min(max_dim,conv_dim * 2 ** (i-1)) #NOTE ou i in STGAN
        dec_in = min(max_dim,conv_dim * 2 ** (i)) #NOTE ou i+1 in STGAN
        enc_size = min(max_dim,conv_dim * 2 ** (i))
        
        if i == n_layers-1: dec_in = enc_size * n_branches
        if i >= n_layers - n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
        if i >= n_layers - 1 - shortcut_layers and i != n_layers-1: # skip connection: n_branches-1 or 1 feature map
            dec_in = dec_in + max(1,n_branches-1)*enc_size 
        if (i==0): dec_out=conv_dim // 4

        dec_layer=[ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias),'relu','batch')]
        if vgg_like and i >= min(3,n_layers - 1 - shortcut_layers):
            dec_layer+=[ConvReluBn(nn.Conv2d(dec_out, dec_out, 3, 1, 1,bias=bias),'relu','batch')]
        decoder.append(nn.Sequential(*dec_layer))

    last_conv = nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=True)
    return decoder, last_conv


class Generator(nn.Module):
    def __init__(self, attr_dim, conv_dim=64, n_layers=5, max_dim=1024, shortcut_layers=2,n_attr_deconv=1, vgg_like=False):
        super(Generator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.n_attr_deconv = n_attr_deconv
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        ##### build encoder
        self.encoder = Encoder(conv_dim,n_layers,max_dim,vgg_like)
        ##### build decoder
        self.decoder, self.last_conv = build_decoder_convs(attr_dim,conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like,n_branches=1)


    #return [encodings,bneck]
    def encode(self,x):
        return self.encoder.encode(x)

    # #bneck is a list of the 3 bnecks, encoder_feats contains the feats from shape/illum
    def decode(self, bneck, a, encodings):
        #expand dimensions of a
        a = a.view((bneck.size(0), self.n_attrs, 1, 1))
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.shortcut_layers:
                #do shortcut connection, not taking the first encoding (mat)
                out = torch.cat([out, encodings[-(i+1)]], dim=1)
            out = dec_layer(self.up(out))
        x = self.last_conv(out)
        x = torch.tanh(x)
        return x

    def forward(self, x, a):
        # propagate encoder layers
        encodings,z = self.encode(x)
        return self.decode(z,a,encodings)

    def split_bneck(self, bneck, do_norm=False):
        self.bneck_shape = bneck.shape
        batch_size = bneck.size(0)
        step = bneck.size(1) // 3

        bneck_material = bneck[:, :step] #.view(batch_size, -1)  # 1/3 of the features
        bneck_shape = bneck[:, step:step * 2]#.view(batch_size, -1)
        bneck_illum = bneck[:, step * 2:]#.view(batch_size, -1)

        if do_norm:
            bneck_material = bneck_material / bneck_material.norm(p=2)
            bneck_shape = bneck_shape / bneck_shape.norm(p=2)
            bneck_illum = bneck_illum / bneck_illum.norm(p=2)

        return bneck_material, bneck_shape, bneck_illum

    def join_bneck(self, bnecks):
        return torch.cat(bnecks, dim=1)#.view(self.bneck_shape)


class GeneratorWithBranches(Generator):
    def __init__(self, attr_dim, conv_dim=64, n_layers=5, max_dim=1024, shortcut_layers=2,n_attr_deconv=1, vgg_like=False):
        super(GeneratorWithBranches, self).__init__(attr_dim, conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like)

        ##### build encoder
        self.encoder_mat = Encoder(conv_dim,n_layers,max_dim,vgg_like)
        self.encoder_illum = Encoder(conv_dim,n_layers,max_dim,vgg_like)
        self.encoder_shape = Encoder(conv_dim,n_layers,max_dim,vgg_like)
        ##### build decoder
        self.decoder, self.last_conv = build_decoder_convs(attr_dim,conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like, n_branches=3)


    #return [encodings,bneck]
    def encode(self,x):
        enc_mat, bneck_mat = self.encoder_mat.encode(x)
        enc_illum, bneck_illum = self.encoder_illum.encode(x)
        enc_shape, bneck_shape = self.encoder_shape.encode(x)
        return [enc_mat,enc_shape,enc_illum],[bneck_mat, bneck_shape, bneck_illum]

    # #bneck is a list of the 3 bnecks, encoder_feats contains the feats from mat/shape/illum
    def decode(self, bnecks, a, encodings):
        bneck=torch.cat(bnecks, dim=1)
        #expand dimensions of a
        a = a.view((bneck.size(0), self.n_attrs, 1, 1))
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.shortcut_layers:
                #do shortcut connection, not taking the first encoding (mat)
                for enc in encodings[1:]:
                    out = torch.cat([out, enc[-(i+1)]], dim=1)
            out = dec_layer(self.up(out))
        x = self.last_conv(out)
        x = torch.tanh(x)
        return x
    #only view them as one dimensional tensors
    def split_bneck(self, bneck, do_norm=False):
        self.bneck_shape = bneck[0].shape
        batch_size = bneck[0].size(0)

        bneck_material = bneck[0]#.view(batch_size, -1)
        bneck_shape = bneck[1]#.view(batch_size, -1)
        bneck_illum = bneck[2]#.view(batch_size, -1)

        if do_norm:
            bneck_material = bneck_material / bneck_material.norm(p=2)
            bneck_shape = bneck_shape / bneck_shape.norm(p=2)
            bneck_illum = bneck_illum / bneck_illum.norm(p=2)

        return bneck_material, bneck_shape, bneck_illum
    #only restore the shape
    def join_bneck(self, bnecks):
        return bnecks
        #return [bneck.view(self.bneck_shape) for bneck in bnecks]



def FC_layers(in_dim,fc_dim,out_dim,tanh):
    layers=[nn.Linear(in_dim, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, out_dim)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Latent_Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, conv_dim=64, fc_dim=1024, n_layers=5, shortcut_layers=2,vgg_like=False,tanh=True):
        super(Latent_Discriminator, self).__init__()
        layers = []
        n_dis_layers = int(np.log2(image_size))
        layers=get_encoder_layers(conv_dim,n_dis_layers, max_dim, norm='batch',dropout=0.3,vgg_like=vgg_like)
        self.conv = nn.Sequential(*layers[n_layers-shortcut_layers:])

        out_conv = min(max_dim,conv_dim * 2 ** (n_dis_layers - 1))
        self.fc_att = FC_layers(out_conv,fc_dim,attr_dim,True)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att


class Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, conv_dim=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        layers=get_encoder_layers(conv_dim,n_layers, max_dim, norm='batch')
        self.conv = nn.Sequential(*layers)

        feature_size = image_size // 2**n_layers
        out_conv = min(max_dim,conv_dim * 2 ** (n_layers - 1))
        self.fc_adv = FC_layers(out_conv * feature_size ** 2,fc_dim,1,False)
        self.fc_att = FC_layers(out_conv * feature_size ** 2,fc_dim,attr_dim,True)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att




if __name__ == '__main__':
    gen = Generator(5, conv_dim=32, n_layers=6, shortcut_layers=0, max_dim=512,vgg_like=True)

    print(gen)
    #summary(gen, [(3, 128, 128), (5,)], device='cpu')

    # dis = Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=7)
    # print(dis)
    # #summary(dis, (3, 128, 128), device='cpu')

    # dis = Latent_Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=6,shortcut_layers=2)
    # print(dis)
