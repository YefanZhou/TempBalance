#!/bin/bash

# SGD baseline   
for line in 24 25 26 27
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# SNR baseline 
for line in 29 30 31 32
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB 
for line in 34 35 36 37
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB + SNR
for line in 39 40 41 42
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done


    



