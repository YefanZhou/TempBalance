# Experiments on Language Modeling



## Install
```bash
bash install.sh
conda activate ww_train_lm
bash penn_tree.sh
```

## Experiments
```bash
export src_path=/home/eecs/yefan0726/ww_train_repos/TempBalance/language_modeling
export ckpt_src_path=/data/yefan0726/checkpoints/tempbalance/lm

# Baseline 
bash ./BTD-Transformer/scripts/tensorized/run_ptb.sh

# TempBalance
bash ./BTD-Transformer/scripts/tensorized/run_ptb_tb.sh
```


## Acknowledgement
1. [Zihang Liu](https://github.com/HenryLiu0820) is the main contributor to the language modeling implementation.
2. We thank the open-sourced codebase [The-compression-of-Transformer](https://github.com/szhangtju/The-compression-of-Transformer/tree/master).