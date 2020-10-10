import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import os

def conv_bn(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, outp):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == outp

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),

                nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outp)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outp)
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, width_multi=1):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            # 480,480,3 -> 240,240,32
            # 240,240,32 -> 240,240,16
            [1, 16, 1, 1],
            # 240,240,16 -> 120,120,24
            [6, 24, 2, 2],
            # 120,120,24 -> 60,60,32
            [6, 32, 3, 2],
            # 60,60,32 -> 30,30,64
            [6, 64, 4, 2],
            # 30,30,64 -> 30,30,96
            [6, 96, 3, 1],
            # 30,30,96 -> 15,15,160
            [6, 160, 3, 2],
            # 15,15,160 -> 15,15,320
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_multi)
        self.last_channel = int(last_channel * width_multi) if width_multi > 1.0 else last_channel

        self.features = [conv_bn(3, input_channel, 2)]
        self.stages = []

        for expand_ratio, output_channel, repeat, stride in interverted_residual_setting:
            output_channel = int(output_channel * width_multi)
            for i in range(repeat):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, expand_ratio))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio))
                input_channel = output_channel

        # self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        # self.features.cuda()
        self.conv2 = self.features[:3]
        self.conv3 = self.features[3:6]
        self.conv4 = self.features[6:13]
        self.conv5 = self.features[13:]
        self._initialize_weights()

    def forward(self, x):
        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv2, conv3, conv4, conv5
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        model.load_state_dict(load_url('http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar'), strict=False)
    return model

if __name__ == "__main__":
    input = torch.rand(1, 3, 480, 480)
    model = MobileNetV2(input_size=480)
    conv2, conv3, conv4, conv5 = model(input)
    print(conv2.size(), conv3.size(), conv4.size(), conv5.size())