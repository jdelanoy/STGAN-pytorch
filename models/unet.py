import torch
import torch.autograd.function as autograd_fn
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


class IntrinsicsSplit(autograd_fn.Function):
    @staticmethod
    def split_bneck(all_feat):
        # skip_features is a tuple of features containing all the features that come from the
        # encoder to the decoder. For instance, the bottleneck but also the skip connections.
        assert isinstance(all_feat, tuple)
        assert len(all_feat) > 0

        # get batch
        batch_size = all_feat[0].size(0)

        # initialize structures to store the splitted features
        feat_sizes = [None] * len(all_feat)
        feat_mater = [None] * len(all_feat)
        feat_shape = [None] * len(all_feat)
        feat_illum = [None] * len(all_feat)

        for i, feat in enumerate(all_feat):
            step = feat.size(1) // 3  # approx. 33% of the features at each layer for each property

            feat_sizes[i] = feat.shape  # we need to store the shape to reconstruct it later

            feat_mater[i] = feat[:, :step].view(batch_size, -1)
            feat_shape[i] = feat[:, step:step * 2].view(batch_size, -1)
            feat_illum[i] = feat[:, step * 2:].view(batch_size, -1)

        return (feat_mater, feat_shape, feat_illum), feat_sizes

    @staticmethod
    def join_bneck(all_feat, all_feat_size):
        concat_feat = [None] * len(all_feat_size)

        for i in range(len(all_feat_size)):
            # concat material (0), geometry (1) and illumination(2) for each set of features
            concat_feat[i] = torch.cat((all_feat[0][i], all_feat[1][i], all_feat[2][i]), dim=1)
            concat_feat[i] = concat_feat[i].view(all_feat_size[i])

        return concat_feat

    @staticmethod
    def forward(ctx, all_features, mode):
        ctx.mode = mode
        all_split_feats, ctx.feat_sizes = IntrinsicsSplit.split_bneck(all_features)
        ctx.feat_mater, ctx.feat_shape, ctx.feat_illum = all_split_feats
        ctx.save_for_backward(all_features)

        return all_features

    @staticmethod
    def backward(ctx, grad_output):
        # initialize all the variables
        clamping_weight = 1 / 100
        mode = ctx.mode
        feat_mater = ctx.bneck_mater
        feat_shape = ctx.bneck_shape
        feat_illum = ctx.bneck_illum

        # we have to return as many gradients as inputs in forward (two in this case).
        # However, the mode does need a gradient thus return None as the second gradient
        bneck_grad = None
        mode_grad = None

        # compute features mean
        feat_mater_mean = feat_mater.mean(dim=0, keepdim=True).expand_as(feat_mater)
        feat_shape_mean = feat_shape.mean(dim=0, keepdim=True).expand_as(feat_shape)
        feat_illum_mean = feat_illum.mean(dim=0, keepdim=True).expand_as(feat_illum)

        # get features gradient for each property
        split_feat_grad, feat_grad_size = IntrinsicsSplit.split_bneck(grad_output)
        bneck_mater_grad, bneck_shape_grad, bneck_illum_grad = split_feat_grad
        print(grad_output)

        # from S3.2 IGN paper: "we train all the neurons which correspond to the inactive
        # transformations with an error gradient equal to their difference from the mean.
        # [...] This regularizing force needs to be scaled to be much smaller than the
        # true training signal, otherwise it can overwhelm the reconstruction goal.
        # Empirically, a factor of 1/100 works well."
        if torch.all(mode == 0):  # only MATERIAL changes in the batch
            bneck_shape_grad = clamping_weight * (feat_shape - feat_shape_mean)
            bneck_illum_grad = clamping_weight * (feat_illum - feat_illum_mean)
        elif torch.all(mode == 1):  # only GEOMETRY changes in the batch
            bneck_mater_grad = clamping_weight * (feat_mater - feat_mater_mean)
            bneck_illum_grad = clamping_weight * (feat_illum - feat_illum_mean)
        elif torch.all(mode == 2):  # only ILLUMINATION changes in the batch
            bneck_shape_grad = clamping_weight * (feat_shape - feat_shape_mean)
            bneck_mater_grad = clamping_weight * (feat_mater - feat_mater_mean)
        else:
            raise ValueError('data sampling mode not understood')

        grad_bneck = (bneck_mater_grad, bneck_shape_grad, bneck_illum_grad)
        grad_bneck = IntrinsicsSplit.join_bneck(grad_bneck, ctx.bneck_size)
        return grad_output, None


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch, act, norm, ign_grad=False):
        super(UNet, self).__init__()
        self.up_and_concat = UpAndConcat()
        self.ign_grad=ign_grad
        
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

        self.split = IntrinsicsSplit()

        self.decoder = nn.ModuleList([
            ConvReluBn(masked_conv3x3(ch[4] + ch[4], ch[4], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[4] + ch[3], ch[3], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[3] + ch[2], ch[2], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[2] + ch[1], ch[1], bias=bias), act, norm),
            ConvReluBn(masked_conv3x3(ch[1] + ch[0], ch[0], bias=bias), act, norm),
        ])

        self.last_conv = masked_conv3x3(ch[0] + in_ch, out_ch, bias=True)

    def forward_encoder(self, x, mode):
        # Encoder
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            x_encoder.insert(0, x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # group all features into a tuple and call the split function to keep track of their
        # special (custom) backward case
        x_encoder, x = self.split_bneck((*x_encoder, x), mode)

        return x, x_encoder

    def split_bneck(self, x, mode):
        x = self.split.apply(x, mode)
        # split again the features in such a way that we return encoder skip connections and bneck
        return x[:-1], x[-1]

    def forward_decoder(self, identity, bneck, encoder_feats):
        # Decoder
        for block, enc_feat in zip(self.decoder, encoder_feats):
            bneck = block(self.up_and_concat(bneck, enc_feat))

        x = self.last_conv(self.up_and_concat(bneck, identity))
        x = torch.tanh(x)
        return x

    def forward(self, x, mode):
        # keep track of the input x
        identity = x

        bneck, enc_feat = self.forward_encoder(x, mode)
        x = self.forward_decoder(identity, bneck, enc_feat)

        return x


class Autoencoder(UNet):
    def __init__(self, in_ch, out_ch, ch, act, norm, ign_grad=False):
        super(Autoencoder, self).__init__(in_ch, out_ch, ch, act, norm, ign_grad)
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

    def get_bneck_split(self):
        return self.split.ctx.bneck_mater, \
               self.split.ctx.bneck_shape, \
               self.split.ctx.bneck_illum


if __name__ == '__main__':
    img = torch.rand(8, 3, 224, 224).cuda()
    mode = torch.ones(8, 1).cuda()

    act = 'leaky_relu'
    norm = 'none'
    model = UNet(in_ch=3, out_ch=3, ch=[64, 128, 256, 512, 512], act=act,
                 norm=norm).cuda()

    print('in shape', img.shape)
    img_hat = model(img, mode).clamp(0, 1)
    print('out shape', img_hat.shape)

    loss = torch.nn.functional.l1_loss(img_hat, img)
    loss.backward()
