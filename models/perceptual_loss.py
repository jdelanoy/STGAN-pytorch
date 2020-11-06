from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@torch.jit.script
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Interpolate(torch.jit.ScriptModule):
    __constants__ = ["size", "mode", "align_corners"]

    def __init__(self, size, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    @torch.jit.script_method
    def forward(self, x):
        return nn.functional.interpolate(input=x,
                                         size=self.size,
                                         mode=self.mode,
                                         align_corners=self.align_corners)


class PerceptualLoss(torch.jit.ScriptModule):
    __constants__ = ["layers", "norm_input", "do_resize", "use_gram_matrix"]

    def __init__(self,
                 layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 requires_grad: bool = False,
                 norm_input: bool = False,
                 do_resize: bool = False,
                 use_gram_matrix: bool = False
                 ):
        super(PerceptualLoss, self).__init__()

        self.layers = layers
        self.model = self.setup_layers(layers)

        self.mse = nn.MSELoss()

        self.norm_input: bool = norm_input
        self.do_resize: bool = do_resize
        self.use_gram_matrix: bool = use_gram_matrix

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.resize = Interpolate(mode='bilinear', size=224, align_corners=False)

        self.mean = torch.nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.jit.script_method
    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        if self.norm_input:
            x_hat = (x_hat - self.mean) / self.std
            x = (x - self.mean) / self.std

        if self.do_resize:
            x_hat = self.resize(x_hat)
            x = self.resize(x)

        if len(x.shape) == 5:  # we have to deal with a transport matrix
            batch, coeffs, channels, width, height = x.shape
            x = x.view(batch * coeffs, channels, width, height)
            x_hat = x_hat.view(batch * coeffs, channels, width, height)

        # make sure that the input is 3 dimensional
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x_hat.size(1) == 1:
            x_hat = x_hat.repeat(1, 3, 1, 1)

        # start computing the loss
        loss: torch.Tensor = torch.tensor([0], device=x.device, dtype=x.dtype)
        for _, layer in self.model.named_children():

            x = layer(x)
            x_hat = layer(x_hat)
            if self.use_gram_matrix:
                loss = loss + self.mse(gram_matrix(x), gram_matrix(x_hat))
            else:
                loss = loss + self.mse(x, x_hat)

        return loss / len(self.layers)

    @staticmethod
    def setup_layers(
            layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
            features: nn.Module = models.vgg16(pretrained=True).features) -> nn.Module:

        # create structure to store the needed layers
        model: nn.Sequential = nn.Sequential()
        for layer in layers:
            model.add_module(layer, nn.Sequential())

        ix_init = 0
        if 'relu1_2' in layers:
            ix_end = 4
            for x in range(ix_init, ix_end):
                curr_module = dict(model.named_children())['relu1_2']
                curr_module.add_module(str(x), features[x])
            ix_init = ix_end
        if 'relu2_2' in layers:
            ix_end = 9
            for x in range(ix_init, ix_end):
                curr_module = dict(model.named_children())['relu2_2']
                curr_module.add_module(str(x), features[x])
            ix_init = ix_end
        if 'relu3_3' in layers:
            ix_end = 16
            for x in range(ix_init, ix_end):
                curr_module = dict(model.named_children())['relu3_3']
                curr_module.add_module(str(x), features[x])
            ix_init = ix_end
        if 'relu4_3' in layers:
            ix_end = 23
            for x in range(ix_init, ix_end):
                curr_module = dict(model.named_children())['relu4_3']
                curr_module.add_module(str(x), features[x])

        return model


if __name__ == '__main__':
    a = torch.randn(8, 3, 256, 256).cuda()
    b = torch.randn(8, 3, 256, 256).cuda()
    loss = PerceptualLoss().cuda()
    print(loss(a, b))
