############### Pytorch CIFAR configuration file ###############
import math
import numpy as np
start_epoch = 1
#[0.44671097, 0.4398105 , 0.4066468 ], std as [0.2603405 , 0.25657743, 0.27126738]
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'tiny-imagenet-200': (0.485, 0.456, 0.406),
    'svhn':     (0.4377, 0.4438, 0.4728)
}
std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'tiny-imagenet-200': (0.229, 0.224, 0.225),
    'svhn':         (0.1980, 0.2010, 0.1970)
}
crop_size = {
    'cifar10': 32,
    'cifar100': 32,
    'svhn': 32,
    'tiny-imagenet-200': 64,
    }
eval_batchsize = {
    'cifar10': 500,
    'cifar100': 500,
    'svhn': 500,
    'tiny-imagenet-200': 200
}

# Only for cifar-10
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def cosine_decay(init, epoch, total_epoch):
    epoch = min(epoch, total_epoch)
    cosine_decay = 0.5 * (1 + math.cos(np.pi * epoch / total_epoch))
    
    return init * cosine_decay




def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
