'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from collections import  OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_0.5': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'VGG16_1.5': [96, 96, 'M', 192, 192, 'M', 384, 384, 384, 'M', 768, 768, 768, 'M', 768, 768, 768, 'M'],
    
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_0.5': [32, 32, 'M', 64,64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'VGG19_1.5': [96, 96, 'M', 192,192, 'M', 384, 384, 384, 384, 'M', 768, 768, 768, 768, 'M', 768, 768, 768, 768, 'M']
}


class VGG_cifar(nn.Module):
    def __init__(self, depth, num_classes, width_factor=1):
        super(VGG_cifar, self).__init__()
        if abs(width_factor - 1) > 1e-4:
            print(f"Initialize the VGG by width factor: {width_factor}")
            self.features = self._make_layers(cfg[f'VGG{depth}_{width_factor}'])
            linear_input = int(round(512 * width_factor))
        else:
            print(f"Initialize the VGG by default width factor 1")
            self.features = self._make_layers(cfg[f'VGG{depth}'])
            linear_input = 512

        self.linear = nn.Linear(linear_input, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        i=0
        for x in cfg:
            i=i+1
            if x == 'M':
                layers.append(('MaxPool2d_{}'.format(i),nn.MaxPool2d(kernel_size=2, stride=2)))
            else:
                layers.append(('conv_{}'.format(i),nn.Conv2d(in_channels, x, kernel_size=3, padding=1)))
                layers.append(('bn_{}'.format(i),nn.BatchNorm2d(x)))
                layers.append(('relu_{}'.format(i), nn.ReLU(inplace=True)))
                in_channels = x
        layers.append(('AvgPool2d_{}'.format(i),nn.AvgPool2d(kernel_size=1, stride=1)))
        return nn.Sequential(OrderedDict(layers))


#test()
