'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1256.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_allwidth(nn.Module):
    def __init__(self, depth, num_classes=10, width_frac=1):
        super(ResNet_allwidth, self).__init__()
        configs = { 18:  (BasicBlock, [2, 2, 2, 2]),
                    34: (BasicBlock,  [3, 4, 6, 3]),
                    50: (Bottleneck,  [3, 4, 6, 3]),
                    101:(Bottleneck,  [3, 4, 23, 3]),
                    152:(Bottleneck,  [3, 8, 36, 3])}
        
        block, num_blocks = configs[depth]

        first = int(round(64 * width_frac))
        # self.in_planes gets updated in self._make_layer()
        self.in_planes = first

        self.conv1 = nn.Conv2d(3, first, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first)
        self.layer1 = self._make_layer(block, first, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, first * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, first * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, first * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(first * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def ResNet18(dataset='cifar100',width_frac=1):
#     num_classes = 100 if dataset=='cifar100' else 10
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_frac=width_frac)

# def ResNet34(dataset='cifar100',width_frac=1):
#     num_classes = 100 if dataset=='cifar100' else 10
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, width_frac=width_frac)

# def ResNet50(dataset='cifar100',width_frac=1):
#     num_classes = 100 if dataset=='cifar100' else 10
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_frac=width_frac)


# def ResNet101(dataset='cifar100',width_frac=1):
#     num_classes = 100 if dataset=='cifar100' else 10
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_frac=width_frac)


# def ResNet152(dataset='cifar100',width_frac=1):
#     num_classes = 100 if dataset=='cifar100' else 10
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, width_frac=width_frac)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()