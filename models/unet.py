import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import activation_func, normalization_func


def conv1x1(inp, out, stride=1, padding=0, bias=False):
    return nn.Conv2d(inp, out, 1, stride, padding, bias=bias)


def conv3x3(inp, out, stride=1, padding=1, bias=False):
    return nn.Conv2d(inp, out, 3, stride, padding, bias=bias)


def conv5x5(inp, out, stride=2, padding=2, bias=False):
    return nn.Conv2d(inp, out, 5, stride, padding, bias=bias)


def conv7x7(inp, out, stride=2, padding=3, bias=False):
    return nn.Conv2d(inp, out, 7, stride, padding, bias=bias)


class UpAndConcat(nn.Module):
    def __init__(self):
        super(UpAndConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        # input is CHW
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]

        # pad the image to allow for arbitrary inputs
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # concat both inputs
        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)
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

    def __init__(self, in_ch, out_ch, act='relu', norm='bn', bias=False):
        super(BottleneckBlock, self).__init__()
        if in_ch != out_ch:
            self.reduction_conv = ConvReluBn(conv3x3(in_ch, out_ch, bias=bias), act, norm)
        else:
            self.reduction_conv = None

        self.activate = activation_func(act)
        self.c1 = conv3x3(in_ch, out_ch, bias=bias)
        self.bn1 = normalization_func(norm)(out_ch)

        self.c2 = conv3x3(out_ch, out_ch, bias=bias)
        self.bn2 = normalization_func(norm)(out_ch)

    def forward(self, x):
        identity = x

        x = self.activate(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        if self.reduction_conv is not None:
            identity = self.reduction_conv(identity)
        x = self.activate(x + identity)
        return x


class IntrinsicsSplit(torch.autograd.Function):
    @staticmethod
    def split_bneck(all_feat):
        # skip_features is a tuple of features containing all the features that come from the
        # encoder to the decoder. For instance, the bottleneck but also the skip connections.
        assert isinstance(all_feat, tuple) or isinstance(all_feat, )
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

        # concat material (0), geometry (1) and illumination(2) for each set of features
        for i in range(len(all_feat_size)):
            concat_feat[i] = torch.cat((all_feat[0][i], all_feat[1][i], all_feat[2][i]), dim=1)
            concat_feat[i] = concat_feat[i].reshape(all_feat_size[i])

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
        print('inbackward!')
        print(grad_output)

        # initialize all the variables
        clamping_weight = 1 / 100
        mode = ctx.mode
        feat_mater = ctx.bneck_mater
        feat_shape = ctx.bneck_shape
        feat_illum = ctx.bneck_illum
        grad_input = grad_output.clone()

        # we have to return as many gradients as inputs in forward (two in this case).
        # However, the mode does need a gradient thus return None as the second gradient
        bneck_grad = None
        mode_grad = None

        # compute features mean
        feat_mater_mean = feat_mater.mean(dim=0, keepdim=True).expand_as(feat_mater)
        feat_shape_mean = feat_shape.mean(dim=0, keepdim=True).expand_as(feat_shape)
        feat_illum_mean = feat_illum.mean(dim=0, keepdim=True).expand_as(feat_illum)

        # get features gradient for each property
        split_feat_grad, feat_grad_size = IntrinsicsSplit.split_bneck(grad_input)
        bneck_mater_grad, bneck_shape_grad, bneck_illum_grad = split_feat_grad

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
        return grad_bneck, None


class Encoder(nn.Module):
    def __init__(self, in_ch, ch, act, norm, ign_grad=False):
        super(Encoder, self).__init__()
        self.ign_grad = ign_grad
        self.split = IntrinsicsSplit()

        bias = norm == 'none'  # use bias only if we do not use a norm layer

        # build the encoder with 3x3 convs with stride 2 (to reduce resolution by 2)
        encoder = [ConvReluBn(conv3x3(in_ch, ch[0], 2, bias=bias, ), act, norm)]
        for i in range(1, len(ch)):
            encoder.append(ConvReluBn(conv3x3(ch[i - 1], ch[i], 2, bias=bias), act, norm))

        self.encoder = nn.ModuleList(encoder)

    def forward(self, x, mode):
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            # keep track of IGN gradient
            if self.ign_grad: x = self.split.apply((x,), mode)[0]
            x_encoder.insert(0, x)

        return x, x_encoder


class Bottleneck(nn.Module):
    def __init__(self, in_ch, act, norm, out_ch=None, bneck_layers=2, ign_grad=False):

        super(Bottleneck, self).__init__()
        self.ign_grad = ign_grad
        self.split = IntrinsicsSplit()

        bias = norm == 'none'  # use bias only if we do not use a norm layer
        if out_ch is None:
            out_ch = in_ch
        # build bottleneck layers
        bneck = [BottleneckBlock(in_ch, out_ch, act, norm, bias=bias) for _ in range(bneck_layers)]
        self.bottleneck = nn.ModuleList(bneck)

    def forward(self, x, mode):
        for block in self.bottleneck:
            x = block(x)
            if self.ign_grad:
                x = self.split.apply((x,), mode)[0]
        return x


class Decoder(nn.Module):
    def __init__(self, ch, out_ch, act, norm):
        super(Decoder, self).__init__()
        bias = norm == 'none'  # use bias only if we do not use a norm layer
        self.up_and_concat = UpAndConcat()  # layer to concat skip connections

        decoder = [ConvReluBn(conv3x3(ch[-1] + ch[-1], ch[-1], bias=bias), act, norm)]
        for i in range(1, len(ch)):
            ix = len(ch) - i - 1
            decoder.append(ConvReluBn(conv3x3(ch[ix + 1] + ch[ix], ch[ix], bias=bias), act, norm))

        self.last_conv = ConvReluBn(conv3x3(ch[0], out_ch, bias=True), act, norm)

        self.decoder = nn.ModuleList(decoder)

    def forward(self, x, x_encoder):
        for block, enc_feat in zip(self.decoder, x_encoder):
            x = block(self.up_and_concat(x, enc_feat))
        x = self.last_conv(self.up_and_concat.up(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch, act, norm, ign_grad=False):
        super(UNet, self).__init__()

        # UNET structure
        self.encoder = Encoder(in_ch, ch, act, norm, ign_grad=ign_grad)
        self.bottleneck = Bottleneck(ch[-1], act, norm, ign_grad=ign_grad)
        self.decoder = Decoder(ch, out_ch, act, norm)

    def forward(self, x, mode):
        x, x_encoder = self.encoder(x, mode)
        x = self.bottleneck(x, mode)
        x = self.decoder(x, x_encoder)

        return torch.tanh(x)


class BranchUNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch, act, norm, bneck_layers=2, ign_grad=False):
        super(BranchUNet, self).__init__()

        self.split = IntrinsicsSplit()

        # UNET structure
        self.encoder_mater = Encoder(in_ch, ch, act, norm, ign_grad=ign_grad)
        self.encoder_shape = Encoder(in_ch, ch, act, norm, ign_grad=ign_grad)
        self.encoder_illum = Encoder(in_ch, ch, act, norm, ign_grad=ign_grad)

        self.bneck_mater = Bottleneck(ch[-1], out_ch=ch[-1], act=act, bneck_layers=bneck_layers,
                                      norm=norm, ign_grad=ign_grad)
        self.bneck_shape = Bottleneck(ch[-1], out_ch=ch[-1], act=act, bneck_layers=bneck_layers,
                                      norm=norm, ign_grad=ign_grad)
        self.bneck_illum = Bottleneck(ch[-1], out_ch=ch[-1], act=act, bneck_layers=bneck_layers,
                                      norm=norm, ign_grad=ign_grad)

        # adapt channels to accommodate all bottlenecks and skipconnections
        # ch[-1] = ch[-1] * 3
        # ch[:-1] = [nch * 2 for nch in ch[:-1]]
        self.up_and_concat = UpAndConcat()  # layer to concat and upscale skip connections
        self.up = self.up_and_concat.up  # upscaling layer

        self.decoder = self.__create_decoder(ch, out_ch, act, norm)
        self.last_conv = ConvReluBn(conv3x3(ch[0], out_ch, bias=True), act, norm)

    def forward(self, x, mode):
        x_encoder_mater, x_encoder_shape, x_encoder_illum = self.forward_encoder(x, mode)
        x_mater = x_encoder_mater.pop(-1)
        x_shape = x_encoder_shape.pop(-1)
        x_illum = x_encoder_illum.pop(-1)

        x_encoder = [torch.cat((x1, x2), dim=1) for x1, x2 in zip(x_encoder_shape, x_encoder_illum)]
        x = torch.cat((x_mater, x_shape, x_illum), dim=1)

        x = self.forward_decoder(x, x_encoder)
        return x

    def forward_encoder(self, x, mode):
        x_mater, x_encoder_mater = self.encoder_mater(x, mode)
        x_shape, x_encoder_shape = self.encoder_shape(x, mode)
        x_illum, x_encoder_illum = self.encoder_illum(x, mode)

        x_encoder_mater.append(self.bneck_mater(x_mater, mode))
        x_encoder_shape.append(self.bneck_shape(x_shape, mode))
        x_encoder_illum.append(self.bneck_illum(x_illum, mode))

        return x_encoder_mater, x_encoder_shape, x_encoder_illum

    def forward_decoder(self, x, x_encoder):
        for block, enc_feat in zip(self.decoder, x_encoder):
            x = block(self.up_and_concat(x, enc_feat))
        x = self.last_conv(x)
        return torch.tanh(x)

    def prepare_features(self, x_encoder_mater, x_encoder_shape, x_encoder_illum):
        x_mater = x_encoder_mater[-1]
        x_shape = x_encoder_shape[-1]
        x_illum = x_encoder_illum[-1]

        x_encoder = [torch.cat((x1, x2), dim=1) for x1, x2 in zip(x_encoder_shape, x_encoder_illum)]
        x = torch.cat((x_mater, x_shape, x_illum), dim=1)

        return x, x_encoder[:-1]

    def __create_decoder(self, ch, out_ch, act, norm):
        bias = norm == 'none'  # use bias only if we do not use a norm layer

        decoder = [ConvReluBn(conv3x3(ch[-1] * 3 + ch[-1] * 2, ch[-1], bias=bias), act, norm)]
        for i in range(1, len(ch)):
            ix = len(ch) - i - 1
            decoder.append(
                ConvReluBn(conv3x3(ch[ix + 1] + ch[ix] * 2, ch[ix], bias=bias), act, norm))

        return nn.ModuleList(decoder)


class Autoencoder(UNet):
    def __init__(self, in_ch, out_ch, ch, act, norm, ign_grad=False):
        super(Autoencoder, self).__init__(in_ch, out_ch, ch, act, norm, ign_grad)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_and_concat = UpAndConcat()

        bias = norm == 'none'  # use bias only if we do not use a norm layer
        self.decoder = nn.ModuleList([
            ConvReluBn(conv3x3(ch[4], ch[3], bias=bias), act, norm),
            ConvReluBn(conv3x3(ch[3], ch[2], bias=bias), act, norm),
            ConvReluBn(conv3x3(ch[2], ch[1], bias=bias), act, norm),
            ConvReluBn(conv3x3(ch[1], ch[0], bias=bias), act, norm),
            ConvReluBn(conv3x3(ch[0], out_ch, bias=bias), act, norm),
        ])
        self.last_conv = conv3x3(out_ch, out_ch, bias=True)

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
    ch = [64, 128, 256]
    model = BranchUNet(in_ch=3, out_ch=3, ch=ch, act=act, norm=norm, ign_grad=False).cuda()

    print('in shape', img.shape)
    img_hat = model(img, mode).clamp(0, 1)
    print('out shape', img_hat.shape)

    loss = torch.nn.functional.l1_loss(img_hat, img)
    loss.backward()
