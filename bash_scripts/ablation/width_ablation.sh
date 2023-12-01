#!/bin/bash


for line in {131..154}
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

