import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, normalize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.normalize = normalize

    def forward(self, input, target):
        self.blocks = self.blocks.to(input)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.normalize:
            input = ((input*0.5+0.5)-self.mean.to(input)) / self.std.to(input)
            target = ((target*0.5+0.5)-self.mean.to(input)) / self.std.to(input)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i,block in enumerate(self.blocks):
            #print(i)
            x = block(x)
            y = block(y)
            #print(x.shape)
            loss = loss + torch.nn.functional.mse_loss(x, y)
        return loss/4
