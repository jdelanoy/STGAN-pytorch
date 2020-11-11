import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


class WithSavedActivations(nn.Module):
    def __init__(self, model, types=(nn.Conv2d, nn.Linear), names=None):
        super(WithSavedActivations, self).__init__()
        self.model = model
        self.activations = {}
        self.detach = True
        self.handles = []
        self.set_keep_layers(types, names)

    def set_keep_layers(self, types=(nn.Conv2d, nn.Linear), names=None):
        for h in self.handles:
            h.remove()

        if names is None:
            for name, layer in self.model.named_modules():
                if isinstance(layer, types):
                    h = layer.register_forward_hook(functools.partial(
                        self._save, name))
                    self.handles.append(h)
        else:
            for name in names:
                layer = layer_by_name(self.model, name)
                h = layer.register_forward_hook(
                    functools.partial(self._save, name))
                self.handles.append(h)

    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def forward(self, input, detach):
        self.detach = detach
        self.activations = {}
        out = self.model(input)
        acts = self.activations
        self.activations = {}
        return out, acts


class ImageNetInputNorm(nn.Module):
    """
    Normalize images channels as torchvision models expects, in a
    differentiable way
    """

    def __init__(self):
        super(ImageNetInputNorm, self).__init__()
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))

        self.register_buffer(
            'norm_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.norm_mean) / self.norm_std

class NoNorm(nn.Module):
    """
    Normalize images channels as torchvision models expects, in a
    differentiable way
    """

    def __init__(self):
        super(NoNorm, self).__init__()

    def forward(self, input):
        return input


def layer_by_name(net, name):
    """
    Get a submodule at any depth of a net by its name
    Args:
        net (nn.Module): the base module containing other modules
        name (str): a name of a submodule of `net`, like `"layer3.0.conv1"`.
    Returns:
        The found layer or `None`
    """
    for l in net.named_modules():
        if l[0] == name:
            return l[1]


def PerceptualNet(layers, use_avg_pool=True):
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
        #'conv3_4', 'relu3_4', 
        'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
        #'conv4_4', 'relu4_4', 
        'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        #'conv5_4', 'relu5_4',  # 'maxpool5'
    ]

    m = OrderedDict()
    l_partial = layers.copy()
    for l_name, l in zip(layer_names, M.vgg16(pretrained=True).eval().features):
        # store only necessary layers
        if len(l_partial) == 0:
            break
        m[l_name] = l
        if l_name in l_partial:
            l_partial.remove(l_name)
    m = nn.Sequential(m)

    for nm, mod in m.named_modules():
        if 'relu' in nm:
            setattr(m, nm, nn.ReLU(False))
        elif 'pool' in nm and use_avg_pool:
            setattr(m, nm, nn.AvgPool2d(2, 2))
    m = WithSavedActivations(m, names=layers)
    return m


class PerceptualLoss(nn.Module):
    def __init__(self, l=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], rescale=False, normalize=False, loss_fn=F.mse_loss):
        super(PerceptualLoss, self).__init__()
        self.m = PerceptualNet(l)
        self.norm = ImageNetInputNorm() if normalize else NoNorm() 
        self.rescale = rescale
        self.loss_fn = loss_fn

    def forward(self, x, y):
        """
        Return the perceptual loss between batch of images `x` and `y`
        """
        if self.rescale:
            y = F.interpolate(y, size=(224, 224), mode='nearest')
            x = F.interpolate(x, size=(224, 224), mode='nearest')

        _, ref = self.m(self.norm(y), detach=True)
        _, acts = self.m(self.norm(x), detach=False)
        loss = 0
        for k in acts.keys():
            loss += self.loss_fn(acts[k], ref[k])

        return loss/len(acts.keys())


if __name__ == '__main__':
    torch.manual_seed(10)
    a = torch.randn(4, 3, 256, 256)
    b = torch.randn(4, 3, 256, 256)

    P_loss = PerceptualLoss(normalize=True)
    print(P_loss(a, b))
