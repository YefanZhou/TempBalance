import os
import torch
from networks import *
from operator import itemgetter
import numpy as np
import math
import tqdm
import time
import pandas as pd
import json

def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)


# Training 
def train(  epoch, 
            net, 
            num_epochs, 
            trainloader, 
            criterion, 
            optimizer, 
            optim_type='SGD', 
            tb_update_interval=0, 
            untuned_lr=0, 
            args=None):
    
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0 
    print(f'Training Epoch {epoch}')
    pbar = tqdm.tqdm(total=len(trainloader), desc="Training")

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()                     # Backward Propagation
        optimizer.step()                    # Optimizer update

        train_loss += loss.item() * targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pbar.update(1)
        
        # if tb_update_interval > 0 and args.total_iters % tb_update_interval == 0:
        #     print(f"--------------------> tb_update_interval: {tb_update_interval}, temp_balance")
        #     temp_balance(args=args, net=net, optimizer=optimizer, epoch=epoch, untuned_lr=untuned_lr, iters=args.total_iters)
        
        # if tb_update_interval > 0:
        #     args.total_iters += 1
            
    pbar.close()
    train_loss /= total
    acc = 100.*correct/total
    acc = acc.item()

    return acc, train_loss


# Testing
def test(epoch, net, testloader, criterion):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        acc = acc.item()
        test_loss = test_loss/total
        
    return acc, test_loss



# Return network & file name
def getNetwork(args, num_classes):
    if args.net_type == 'vgg_cifar':
        net = VGG_cifar(args.depth,  num_classes, args.widen_factor)
        file_name = 'vgg_cifar'
    elif args.net_type == 'resnet':
        net = ResNet(args.depth,  num_classes, args.widen_factor)
        file_name = 'resnet'
    elif args.net_type == 'resnet_tiny_imagenet':
        net = ResNet_tiny_imagenet(args.depth, num_classes=num_classes)
        file_name = 'resnet_tiny_imagenet'
    elif args.net_type == 'wide_resnet':
        net = Wide_ResNet(depth=args.depth, 
                            widen_factor=args.widen_factor, 
                            num_classes=num_classes)
        file_name = 'wide_resnet'
        
    return net, file_name



def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)
