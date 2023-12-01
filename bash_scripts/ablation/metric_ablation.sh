#!/bin/bash


for line in {158..169}
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done