from __future__ import print_function
import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from os.path import join
from tempbalance import Tempbalance
from sgdsnr import SGDSNR
from adamp import SGDP, AdamP
import config as cf
import torch_optimizer
from lars_optim import LARS, LAMB
from utils import train, test, getNetwork, save_args_to_file


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr',             type=float,      default=0.01,                         help='learning_rate')
parser.add_argument('--net-type',       type=str,        default='wide-resnet',                help='model')
parser.add_argument('--depth',          type=int,        default=28,                           help='depth of model')
parser.add_argument('--num-epochs',     type=int,        default=200,                          help='number of epochs')
parser.add_argument('--widen-factor',   type=float,      default=1,                           help='width of model')
parser.add_argument('--dataset',        type=str,        default='cifar10',                    help='dataset = [cifar10/cifar100]')
parser.add_argument('--lr-sche',        type=str,        default='cosine',                       choices=['cosine'])
parser.add_argument('--weight-decay',   type=float,      default=1e-4) # 5e-4
parser.add_argument('--ckpt-path',      type=str,        default='',                            help='path to checkpoints')
parser.add_argument('--print-tofile',   default=False,  type=lambda x: (str(x).lower() == 'true'), help='print to file')

parser.add_argument('--batch-size',   type=int,          default=128) # 5e-4
parser.add_argument('--datadir',        type=str,        default='',                            help='directory of dataset')
parser.add_argument('--optim-type',     type=str,        default='SGD',                        help='type of optimizer')
parser.add_argument('--resume',         type=str,        default='',                           help='resume from checkpoint')
parser.add_argument('--seed',           type=int,        default=42) 
parser.add_argument('--ww-interval',    type=int,        default=1)
parser.add_argument('--epochs-to-save',  type=int,       nargs='+',  default=[])
parser.add_argument('--pl-fitting',     type=str,        default='median', choices=['median', 'goodness-of-fit', 'fix-finger'])

# temperature balance related 
parser.add_argument('--use-tb',             default=True,  type=lambda x: (str(x).lower() == 'true'), help='use temp balance')
parser.add_argument('--remove-last-layer',  default=True,   type=lambda x: (str(x).lower() == 'true'),  help='if remove the last layer')
parser.add_argument('--remove-first-layer', default=True,   type=lambda x: (str(x).lower() == 'true'),  help='if remove the first layer')
parser.add_argument('--batchnorm',          default=True,   type=lambda x: (str(x).lower() == 'true'),  help='balancing batch norm layer')
parser.add_argument('--filter-zeros',       default=False,  type=lambda x: (str(x).lower() == 'true')   )
parser.add_argument('--esd-metric-for-tb',   type=str,      default='alpha',  help='ww metric')
parser.add_argument('--assign-func',         type=str,        default='',       help='assignment function for layerwise lr')
parser.add_argument('--lr-min-ratio',        type=float,    default=0.5)
parser.add_argument('--lr-max-ratio',        type=float,    default=1.5)
parser.add_argument('--xmin-pos',            type=float,    default=2, help='xmin_index = size of eigs // xmin_pos')
parser.add_argument('--batchnorm-type',      type=str,      default='name',  help='method to change batchnorm layer learning rate')
parser.add_argument('--look-k',              type=int,      default=5,        help='')
parser.add_argument('--look-alpha',          type=float,    default=0.8,       help='')
parser.add_argument('--T_0',                 type=int,      default=10,       help='')
parser.add_argument('--T-mult',              type=int,      default=2,       help='')

# spectral regularization related
parser.add_argument('--sg',                 type=float, default=0.01, help='spectrum regularization')


args = parser.parse_args()

print(args)
# Save the arguments to a file
save_args_to_file(args, join(args.ckpt_path, 'args.json'))

def set_seed(seed=42):
    print(f"=====> Set the random seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = cf.start_epoch
set_seed(args.seed)


# Data Loader
print('\n[Phase 1] : Data Preparation')
print(f"prepare preprocessing, {args.dataset}")

transform_train = transforms.Compose([
    transforms.RandomCrop(cf.crop_size[args.dataset], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


data_path = join(args.datadir, args.dataset)
if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, 
                                            download=True, 
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, 
                                            download=False, 
                                            transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, 
                                                download=True, 
                                                transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, 
                                                download=False, 
                                                transform=transform_test)
    num_classes = 100

elif(args.dataset == 'svhn'):
    print("| Preparing SVHN dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.SVHN(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=transform_train)
    testset = torchvision.datasets.SVHN(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=transform_test)
    num_classes = 10
    
elif(args.dataset == 'tiny-imagenet-200'): 
    print("| Preparing tiny-imagenet-200 dataset...")
    sys.stdout.write("| ")
    trainset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform_train)
    testset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform_test)
    num_classes = 200
else:
    raise NotImplementedError
    
trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=6)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=cf.eval_batchsize[args.dataset], 
                                            shuffle=False, 
                                            num_workers=4)

Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)

if args.print_tofile:
    # Open files for stdout and stderr redirection
    stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
    stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
    # Redirect stdout and stderr to the files
    sys.stdout = stdout_file
    sys.stderr = stderr_file
    

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    net, file_name = getNetwork(args, num_classes)
    checkpoint = torch.load(args.resume, map_location='cpu')
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['test_acc']
    start_epoch = checkpoint['epoch']
    print(f"Loaded Epoch: {start_epoch} \n Test Acc: {best_acc:.3f} Train Acc: {checkpoint['train_acc']:.3f}")
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args, num_classes)
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    best_acc = 0

if use_cuda:
    net.cuda()
    cudnn.benchmark = True
    
criterion = nn.CrossEntropyLoss()
print(net)

if args.use_tb:
    print("##############Enable and init Temp Balancing##################")
    tb_scheduler = Tempbalance(net=net, 
                    pl_fitting=args.pl_fitting,
                    xmin_pos=args.xmin_pos, 
                    filter_zeros=args.filter_zeros,
                    remove_first_layer=args.remove_first_layer,
                    remove_last_layer=args.remove_last_layer,
                    esd_metric_for_tb=args.esd_metric_for_tb,
                    assign_func=args.assign_func,
                    lr_min_ratio=args.lr_min_ratio,
                    lr_max_ratio=args.lr_max_ratio,
                    batchnorm=args.batchnorm,
                    batchnorm_type=args.batchnorm_type
                    )

    tb_param_group, _ = \
        tb_scheduler.build_optimizer_param_group(untuned_lr=args.lr, initialize=True)
    
    if args.optim_type == 'SGD':
        optimizer = optim.SGD(tb_param_group,
                        momentum=0.9, 
                        weight_decay=args.weight_decay)
    elif args.optim_type == 'SGDSNR':
        optimizer = SGDSNR(tb_param_group, 
                        momentum=0.9, 
                        weight_decay=args.weight_decay, 
                        spectrum_regularization=args.sg)
    elif args.optim_type == 'SGDP':
        optimizer = SGDP(tb_param_group, 
                        momentum=0.9, 
                        weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
else:
    print('Disable Temp Balancing')
    if args.optim_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), 
                        lr=args.lr,  
                        momentum=0.9, 
                        weight_decay=args.weight_decay)
    elif args.optim_type == 'SGDSNR':
        optimizer = SGDSNR(net.parameters(), 
                        lr=args.lr,
                        momentum=0.9, 
                        weight_decay=args.weight_decay, 
                        spectrum_regularization=args.sg)
    elif args.optim_type == 'SGDP':
        optimizer = SGDP( net.parameters(), 
                        lr=args.lr,  
                        momentum=0.9, 
                        weight_decay=args.weight_decay)
    elif args.optim_type == 'AdamP':
        optimizer = AdamP( net.parameters(), 
                        lr=args.lr,  
                        betas=(0.9, 0.999),
                        weight_decay=args.weight_decay)
        
    elif args.optim_type == 'LARS':
        optimizer = LARS(net.parameters(), 
                            lr=args.lr,  
                            weight_decay=args.weight_decay)
    
    elif args.optim_type == 'LAMB':
        optimizer = LAMB(net.parameters(), 
                            lr=args.lr,  
                            weight_decay=args.weight_decay) 
        
    elif args.optim_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), 
                            lr=args.lr,  
                            weight_decay=args.weight_decay) 
        
    elif args.optim_type == 'Lookahead':
        optimizer = optim.SGD(net.parameters(), 
                                lr=args.lr,  
                                momentum=0.9, 
                                weight_decay=args.weight_decay)
        optimizer = torch_optimizer.Lookahead(optimizer, 
                                                k=args.look_k, 
                                                alpha=args.look_alpha)
    elif args.optim_type == 'SGDR':
        optimizer = optim.SGD(net.parameters(), 
                                lr=args.lr,  
                                momentum=0.9, 
                                weight_decay=args.weight_decay)
        scheduler = \
            optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                            T_0=args.T_0, 
                                                            T_mult=args.T_mult)
    else:
        raise NotImplementedError
    
# set a basic global learning rate scheduler
if args.lr_sche == 'cosine':
    lr_schedule = cf.cosine_decay
else:
    raise NotImplementedError

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

test_acc, test_loss = test(epoch=0, net=net, testloader=testloader, criterion=criterion)
print(f"Reevaluated: Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}")


elapsed_time = 0
training_stats = \
{'test_acc': [test_acc],
'test_loss': [test_loss],
'train_acc': [],
'train_loss': [],
'current_lr':[],
'schedule_next_lr':[],
'epoch_time':[],
'elapsed_time':[]
}


untuned_lr = args.lr
is_current_best=False
for epoch in range(start_epoch, start_epoch+args.num_epochs):
    epoch_start_time = time.time()

    # this is current LR
    current_lr = untuned_lr
    train_acc, train_loss = \
                train(epoch, net, args.num_epochs, trainloader, criterion, optimizer)
    print("\n| Train Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, train_loss, train_acc))
    test_acc, test_loss = \
                test(epoch, net, testloader, criterion)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, test_acc))

    # save in interval
    if epoch in args.epochs_to_save:
        state = {
            'net': net.state_dict(),
            'test_acc':test_acc,
            'test_loss':test_loss,
            'train_acc':train_acc,
            'train_loss':train_loss,
            'epoch':epoch
        }
        torch.save(state, join(args.ckpt_path, f'epoch_{epoch}.ckpt'))
    
    # save best
    if test_acc > best_acc:
        print('| Saving Best model')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'test_acc':test_acc,
            'best_acc': best_acc,
            'test_loss':test_loss,
            'train_acc':train_acc,
            'train_loss':train_loss,
            'epoch':epoch
        }
        best_acc = test_acc
        is_current_best=True
        torch.save(state, join(args.ckpt_path, f'epoch_best.ckpt'))
    else:
        is_current_best=False
    
    
    untuned_lr = \
        lr_schedule(args.lr, epoch, args.num_epochs)
    
    if args.use_tb:
        print('----> One step of Temp Balancing')
        tb_scheduler.step(optimizer, untuned_lr)
    else:
        # scheduling by default or some schedulers 
        if args.optim_type == 'SGDR':
            print('lr scheduled by SGDR')
            scheduler.step()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = untuned_lr
            
    epoch_time = time.time() - epoch_start_time
    elapsed_time += epoch_time
    
    training_stats['test_acc'].append(test_acc)
    training_stats['test_loss'].append(test_loss)
    training_stats['train_acc'].append(train_acc)
    training_stats['train_loss'].append(train_loss)
    training_stats['current_lr'].append(current_lr)
    training_stats['schedule_next_lr'].append(untuned_lr)
    training_stats['epoch_time'].append(epoch_time)
    training_stats['elapsed_time'].append(elapsed_time)
    
    np.save(join(args.ckpt_path, "training_stats.npy"), training_stats)
    
    
if args.print_tofile:
    # Close the files to flush the output
    stdout_file.close()
    stderr_file.close()