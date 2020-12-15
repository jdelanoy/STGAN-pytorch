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
    bias = norm == 'none'  # use bias only if we do not use a norm layer

    layers = []
    in_channels = 3
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer=[ConvReluBn(nn.Conv2d(in_channels, out_channels, 4, 2, 1,bias=bias),'leaky_relu',norm)]
        if (vgg_like and i >= 3 and i<n_layers-1):
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),'leaky_relu',norm)]
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),'leaky_relu',norm)]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers


ch=[64, 128, 256, 512, 512]


class Encoder(nn.Module):
    def __init__(self, conv_dim, n_layers, max_dim, vgg_like):
        super(Encoder, self).__init__()
        enc_layers=get_encoder_layers(conv_dim,n_layers,max_dim,norm='batch',vgg_like=vgg_like) #NOTE bias=false for STGAN
        self.encoder = nn.ModuleList(enc_layers)

    #return [encodings,bneck]
    def encode(self,x):
        # propagate encoder layers
        encoded = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            encoded.append(x_)
        return encoded, encoded[-1]


class EncoderManu(nn.Module):
    def __init__(self, act,norm):
        super(EncoderManu, self).__init__()
        bias = norm == 'none'
        self.encoder = nn.ModuleList([
            ConvReluBn(masked_conv7x7(3, ch[0], bias=bias), act, norm),
            ConvReluBn(masked_conv5x5(ch[0], ch[1], bias=bias), act, norm),
            ConvReluBn(masked_conv5x5(ch[1], ch[2], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[2], ch[3], bias=bias, stride=2), act, norm),
            ConvReluBn(masked_conv3x3(ch[3], ch[4], bias=bias, stride=2), act, norm),
        ])

        self.bottleneck = nn.ModuleList([
            BottleneckBlock(ch[4], ch[4], act, norm, bias=bias),
            BottleneckBlock(ch[4], ch[4], act, norm, bias=bias),
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


def build_decoder_convs_manu(act,norm):
    bias = norm == 'none'
    decoder = nn.ModuleList([
        ConvReluBn(masked_conv3x3(ch[4]+1, ch[3], bias=bias), act, norm),
        ConvReluBn(masked_conv3x3(ch[3], ch[2], bias=bias), act, norm),
        ConvReluBn(masked_conv3x3(ch[2], ch[1], bias=bias), act, norm),
        ConvReluBn(masked_conv3x3(ch[1], ch[0], bias=bias), act, norm),
        ConvReluBn(masked_conv3x3(ch[0], 3, bias=bias), act, norm),
    ])
    last_conv = masked_conv3x3(3, 3, bias=True)
    return decoder, last_conv

def build_decoder_convs(attr_dim,conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like):
    bias = norm == 'none'
    decoder = nn.ModuleList()
    for i in reversed(range(n_layers)):
        #size if inputs/outputs
        dec_out = min(max_dim,conv_dim * 2 ** (i-1)) #NOTE ou i in STGAN
        dec_in = min(max_dim,conv_dim * 2 ** (i)) #NOTE ou i+1 in STGAN
        enc_size = min(max_dim,conv_dim * 2 ** (i))

        if i >= n_layers - n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
        if i >= n_layers - 1 - shortcut_layers and i != n_layers-1: # skip connection
            dec_in = dec_in + enc_size
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

        # ##### build encoder
        # self.encoder = Encoder(conv_dim,n_layers,max_dim,vgg_like)
        # ##### build decoder
        # self.decoder, self.last_conv = build_decoder_convs(attr_dim,conv_dim, n_layers, max_dim, shortcut_layers,n_attr_deconv, vgg_like)


        ############## archi from Manu
        act='relu'
        norm='batch'
        bias = norm == 'none'
        self.encoder = EncoderManu(act,norm)
        #self.encoder_illum = EncoderManu(act,norm)
        #self.encoder_shape = EncoderManu(act,norm)
        self.decoder, self.last_conv = build_decoder_convs_manu(act,norm)


    #return [encodings,bneck]
    def encode(self,x):
        return self.encoder.encode(x)
        enc_mat, bneck_mat = self.encoder_mat.encode(x)
        enc_illum, bneck_illum = self.encoder_illum.encode(x)
        enc_shape, bneck_shape = self.encoder_shape.encode(x)
        return [enc_shape,enc_illum],[bneck_mat, bneck_shape, bneck_illum]

    # #bneck is a list of the 3 bnecks, encoder_feats contains the feats from shape/illum
    # def decode(self, bnecks, a, encoder_feats):
    #     bneck=torch.cat(bnecks, dim=1)
    #     # Decoder
    #     for i, block in enumerate(self.decoder):
    #         bneck = block(self.up(bneck))

    #     x = self.last_conv(bneck)
    #     x = torch.tanh(x)
    #     return x


    def decode(self, bnecks, a, encodings):
        bneck=bnecks #torch.cat(bnecks, dim=1)
        #expand dimensions of a
        a = a.view((bnecks.size(0), self.n_attrs, 1, 1))
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if 0 < i <= self.shortcut_layers:
                #do shortcut connection
                out = torch.cat([out, encodings[-(i+1)]], dim=1)
            out = dec_layer(self.up(out))
        x = self.last_conv(out)
        x = torch.tanh(x)
        return out

    def forward(self, x, a):
        # propagate encoder layers
        encodings,z = self.encode(x)
        return self.decode(z,a,encodings)


class DisentangledGenerator(nn.Module):
    def __init__(self, attr_dim, n_embeddings, conv_dim=64, n_layers=5, max_dim=1024, shortcut_layers=2, stu_kernel_size=3, use_stu=True, one_more_conv=True, n_attr_deconv=1, vgg_like=False, image_size=128, fc_dim=256):
        super(DisentangledGenerator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu
        self.n_attr_deconv = n_attr_deconv

        ##### build encoder
        enc_layers=get_encoder_layers(conv_dim>>1,n_layers,max_dim>>1,bias=True,vgg_like=vgg_like) #NOTE bias=false for STGAN
        self.encoder = nn.ModuleList(enc_layers)

        feature_size = image_size // 2**n_layers
        out_conv = min(max_dim>>1,(conv_dim>>1) * 2 ** (n_layers - 1))

        self.features = nn.Sequential(
            nn.Linear(out_conv * feature_size ** 2, fc_dim*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.stu = nn.ModuleList()
        ##### build decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(fc_dim*2*(n_embeddings+1),fc_dim*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.n_layers+1)):
            #size if inputs/outputs
            dec_out = min(max_dim,conv_dim * 2 ** (i-1)) #NOTE ou i in STGAN
            dec_in = min(max_dim,conv_dim * 2 ** (i)) #NOTE ou i+1 in STGAN
            enc_size = min(max_dim,conv_dim * 2 ** (i))>>1

            #if i == self.n_layers-1: dec_in = enc_size
            if i >= self.n_layers - self.n_attr_deconv + 1: dec_in = dec_in + attr_dim #concatenate attribute
            if i >= self.n_layers - self.shortcut_layers: # and i != self.n_layers-1: # skip connection 
                dec_in = dec_in + n_embeddings * (enc_size)
                if use_stu:
                    self.stu.append(ConvGRUCell(self.n_attrs, enc_size, enc_size, min(max_dim,enc_size*2), stu_kernel_size))
            print(i,dec_in,dec_out,enc_size)

            if i > 0:
                if vgg_like and i >= min(3,self.n_layers - 1 - self.shortcut_layers):
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias),'relu','batch'),
                        ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias),'relu','batch')
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias),'relu','batch')
                    ))
            else: #last layer
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        ConvReluBn(nn.Conv2d(dec_in, conv_dim // 4, 3, 1, 1,bias=bias),'relu','batch'),
                        nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(dec_in, 3, 3, 1, 1),
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

    def decode_from_disentangled(self, z, a, encodings):
        #concat all features
        out = z
        for enc in encodings:
            out = torch.cat([out, enc[-1]], dim=1)
        out=self.decoder_fc(out)
        out = out.unsqueeze(-1).unsqueeze(-1) # expand(out.size()[0], out.size()[1],1,1)
        stu_state = z
        a = a.view((out.size(0), self.n_attrs, 1, 1))

        for i, dec_layer in enumerate(self.decoder):
            #print(i)
            if i < self.n_attr_deconv:
                #concatenate attribute
                size = out.size(2)
                attr = a.expand((out.size(0), self.n_attrs, size, size))
                out = torch.cat([out, attr], dim=1)
            if  0 < i <= self.shortcut_layers:
                #do shortcut connection
                for enc in encodings:
                    #print(out.shape,enc[self.n_layers-1-i].shape)
                    out = torch.cat([out, enc[self.n_layers-i]], dim=1)
                    #print(out.shape)
            out = dec_layer(out)
        return out

    def encode(self, x):
        # propagate encoder layers
        encoded = []
        y = x
        for layer in self.encoder:
            y = layer(y)
            encoded.append(y)
        y = y.view(y.size()[0], -1)
        y=self.features(y)
        encoded.append(y)
        return encoded

    def forward(self, x, a, encodings):
        # propagate encoder layers
        encoded = self.encode(x)
        out=self.decode_from_disentangled(encoded[-1],a,encodings)
        return out,encoded

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
