#!/bin/bash

# CAL
for line in 72 73
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# SGDR
for line in 76 77
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_moreopt.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# LARS
for line in 80 81
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# Lookahead
for line in 84 85
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_moreopt.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# SGDP
for line in 88 89
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB
for line in 92 93
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# TB + SGDP
for line in 96 97
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done

# LAMB
for line in 100 101
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train_tb.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done