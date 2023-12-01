############### Pytorch CIFAR configuration file ###############
import math
import numpy as np
start_epoch = 1
#num_epochs = 200
batch_size = 128
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'svhn':( 0.4377, 0.4438, 0.4728)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'svhn':(0.1980, 0.2010, 0.1970)
}

# Only for cifar-10
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def stepwise_decay(init, epoch, total_epoch, warmup_epochs=0):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def cosine_decay(init, epoch, total_epoch, warmup_epochs=0):
    epoch = min(epoch, total_epoch)
    cosine_decay = 0.5 * (1 + math.cos(np.pi * epoch / total_epoch))
    
    return init * cosine_decay



#https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py#L358
def warmup_cosine_decay(max_lr, 
                        epoch, 
                        total_epoch, 
                        warmup_epochs=0, 
                        min_lr=1e-6, 
                        start_warmup_lr=0):
    if epoch <= warmup_epochs:
        return (max_lr - start_warmup_lr) * epoch / warmup_epochs + start_warmup_lr
    else:
        curr_lr =  min_lr + (max_lr - min_lr) \
            * (1 + math.cos(math.pi * (epoch - warmup_epochs) \
                            / (total_epoch - warmup_epochs))  ) / 2
        
        if math.sin(math.pi * (epoch - warmup_epochs) \
                            / (total_epoch - warmup_epochs)) > 0:
            return curr_lr
        else:
            return min_lr



def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
