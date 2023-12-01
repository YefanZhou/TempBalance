#!/bin/bash

## download the TIN dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
mkdir -p ${data_path}
mv tiny-imagenet-200.zip ${data_path}
cp tiny_imagenet_setup.sh ${data_path}
cd ${data_path}
unzip tiny-imagenet-200.zip
chmod +x tiny_imagenet_setup.sh
./tiny_imagenet_setup.sh