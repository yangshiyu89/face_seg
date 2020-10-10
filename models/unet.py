import torch
from torch import nn
from torch.nn import functional as F
import os
import sys
sys.path.append('./')
from models import backbones

class Unet(nn.Module):
    def __init__(self, n_classes=9, expand_ratio=6, pretrained=True):
        super(Unet, self).__init__()
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.conv4 = nn.Sequential(
            nn.Conv2d(320 + 96, 96 * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96 * expand_ratio),
            nn.ReLU6(inplace=True), 

            nn.Conv2d(96 * expand_ratio, 96 * expand_ratio, 3, 1, 1, groups=96 * expand_ratio, bias=False),
            nn.BatchNorm2d(96 * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(96 * expand_ratio, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96 + 32, 32 * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32 * expand_ratio),
            nn.ReLU6(inplace=True), 

            nn.Conv2d(32 * expand_ratio, 32 * expand_ratio, 3, 1, 1, groups=32 * expand_ratio, bias=False),
            nn.BatchNorm2d(32 * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(32 * expand_ratio, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),

        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + 24, 24 * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24 * expand_ratio),
            nn.ReLU6(inplace=True), 

            nn.Conv2d(24 * expand_ratio, 24 * expand_ratio, 3, 1, 1, groups=24 * expand_ratio, bias=False),
            nn.BatchNorm2d(24 * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(24 * expand_ratio, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24),
            
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(24, 16 * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16 * expand_ratio),
            nn.ReLU6(inplace=True), 

            nn.Conv2d(16 * expand_ratio, 16 * expand_ratio, 3, 1, 1, groups=16 * expand_ratio, bias=False),
            nn.BatchNorm2d(16 * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(16 * expand_ratio, self.n_classes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_classes),
        )

    def forward(self, x):
        model = backbones.mobilenetv2.MobileNetV2(input_size=480)
        if self.pretrained:
            model.load_state_dict(backbones.mobilenetv2.load_url('http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar'), strict=False)
        model.cuda()
        conv2, conv3, conv4, conv5 = model(x)
        up4 = F.interpolate(conv5, size=conv4.size()[2:], mode='bilinear', align_corners=True)
        up4 = torch.cat((up4, conv4), dim=1)
        up4 = self.conv4(up4)

        up3 = F.interpolate(up4, size=conv3.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat((up3, conv3), dim=1)
        up3 = self.conv3(up3)

        up2 = F.interpolate(up3, size=conv2.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat((up2, conv2), dim=1)
        up2 = self.conv2(up2)

        x = F.interpolate(up2, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv1(x)

        return x

if __name__ == "__main__":
    input = torch.rand(2, 3, 480, 480)
    model = Unet()
    output = model(input)
    print(output.size())