import torch
import torch.nn as nn
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

    def __init__(self, in_ch, out_ch, activation='relu', normalization='batch', bias=False):
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
