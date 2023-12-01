import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
import sys

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=np.sqrt(2))
#         init.constant(m.bias, 0)
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return input1 + input2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.shortcut = nn.Sequential()
        ## remove the batch norm layer in shortcuts according to 
        # https://github.com/google-research/google-research/blob/master/do_wide_and_deep_networks_learn_the_same_things/resnet_cifar.py
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True)
                #nn.BatchNorm2d(self.expansion*planes)
            )

        self.bn2 = nn.BatchNorm2d(planes)
        self.add = Add()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, width_multiplier=1):
        super(ResNet, self).__init__()
        
        assert ((depth-2)%6 ==0), 'resnet depth should be 6n+2'
        
        k = width_multiplier
        block = BasicBlock
        num_block = (depth-2) // 6

        nStages = [16*k, 16*k, 32*k, 64*k]
        self.in_planes = nStages[0]
        self.conv1 = conv3x3(3, nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nStages[1], num_block, stride=1)
        self.layer2 = self._make_layer(block, nStages[2], num_block, stride=2)
        self.layer3 = self._make_layer(block, nStages[3], num_block, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(nStages[3]*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # linear layer
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out) #F.avg_pool2d(out, 8)
        out = self.flatten(out)
        #out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())


#class Bottleneck(nn.Module):
    # expansion = 4

    # def __init__(self, in_planes, planes, stride=1):
    #     super(Bottleneck, self).__init__()
    #     self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
    #     self.bn1 = nn.BatchNorm2d(planes)
    #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
    #     self.bn2 = nn.BatchNorm2d(planes)
    #     self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
    #     self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    #     self.shortcut = nn.Sequential()
    #     if stride != 1 or in_planes != self.expansion*planes:
    #         self.shortcut = nn.Sequential(
    #             nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
    #             nn.BatchNorm2d(self.expansion*planes)
    #         )

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = F.relu(self.bn2(self.conv2(out)))
    #     out = self.bn3(self.conv3(out))
    #     out += self.shortcut(x)
    #     out = F.relu(out)

    #     return out