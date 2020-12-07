import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvReluBn(nn.Module):
    def __init__(self, conv_layer, activation='relu', normalization='bn'):
        super(ConvReluBn, self).__init__()
        self.conv = conv_layer
        self.bn = normalization_func(normalization)(self.conv.out_channels)
        self.activate = activation_func(activation)

    def forward(self, x):
        x = self.activate(self.bn(self.conv(x)))
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


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch, act, norm):
        super(UNet, self).__init__()
        self.up_and_concat = UpAndConcat()

        # TODO: iterate over ch and automatically create the encoder and decoder
        assert len(ch) == 5

        bias = norm == 'none'  # use bias only if we do not use a norm layer

        self.encoder = nn.ModuleList([
            ConvReluBn(masked_conv7x7(in_ch, ch[0], bias=bias), act, norm),
            ConvReluBn(masked_conv5x5(ch[0], ch[1], bias=bias), act, norm),
            ConvReluBn(masked_conv5x5(ch[1], ch[2], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[2], ch[3], bias=bias, stride=2), act, norm),
            ConvReluBn(masked_conv3x3(ch[3], ch[4], bias=bias, stride=2), act, norm),
        ])

        self.bottleneck = nn.ModuleList([
            BottleneckBlock(ch[4], ch[4], act, norm, bias=bias),
            BottleneckBlock(ch[4], ch[4], act, norm, bias=bias),
        ])
        self.decoder = nn.ModuleList([
            ConvReluBn(masked_conv3x3(ch[4] + ch[4], ch[4], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[4] + ch[3], ch[3], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[3] + ch[2], ch[2], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[2] + ch[1], ch[1], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[1] + ch[0], ch[0], bias=bias), act, norm),
        ])

        self.last_conv = masked_conv3x3(ch[0] + in_ch, out_ch, bias=True)

    def forward_encoder(self, x):
        # Encoder
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            x_encoder.insert(0, x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        return x, x_encoder

    def forward_decoder(self, identity, bneck, encoder_feats):
        # Decoder
        for block, enc_feat in zip(self.decoder, encoder_feats):
            bneck = block(self.up_and_concat(bneck, enc_feat))

        x = self.last_conv(self.up_and_concat(bneck, identity))
        x = torch.tanh(x)
        return x

    def forward(self, x):

        # keep track of the input x
        identity = x

        bneck, enc_feat = self.forward_encoder(x)
        x = self.forward_decoder(identity, bneck, enc_feat)

        return x


class Autoencoder(UNet):
    def __init__(self, in_ch, out_ch, ch, act, norm):
        super(Autoencoder, self).__init__(in_ch, out_ch, ch, act, norm)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_and_concat = UpAndConcat()

        bias = norm == 'none'  # use bias only if we do not use a norm layer
        self.decoder = nn.ModuleList([
            ConvReluBn(masked_conv3x3(ch[4], ch[3], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[3], ch[2], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[2], ch[1], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[1], ch[0], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[0], out_ch, bias=bias), act, norm),
        ])
        self.last_conv = masked_conv3x3(out_ch, out_ch, bias=True)

    def forward_decoder(self, identity, bneck, encoder_feats):
        # Decoder
        for i, block in enumerate(self.decoder):
            bneck = block(self.up(bneck))

        x = self.last_conv(bneck)
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    img = torch.rand(8, 3, 224, 224).cuda()
    act = 'leaky_relu'
    norm = 'none'
    model = Autoencoder(in_ch=3, out_ch=3, ch=[64, 128, 256, 512, 512], act=act,
                        norm=norm).cuda()

    print('in shape', img.shape)
    x = model(img).clamp(0, 1)
    print('out shape', x.shape)
