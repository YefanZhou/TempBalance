#!/bin/bash

# SGD baseline   
for line in 58 59
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# SNR baseline 
for line in 61 62
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB 
for line in 64 65
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB + SNR
for line in 67 68
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done


    



