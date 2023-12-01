pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww_train_lm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

cd ${src_path}/BTD-Transformer/

for SLURM_ARRAY_TASK_ID in {22..26..1}
    do 
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/BTD-Transformer/scripts/txt/ptb_baseline.txt)

        lr=$(echo $cfg | cut -f 1 -d ' ')
        seed=$(echo $cfg | cut -f 2 -d ' ')
        max_epoch=$(echo $cfg | cut -f 3 -d ' ')
        esd_interval=$(echo $cfg | cut -f 4 -d ' ')
        optim=$(echo $cfg | cut -f 5 -d ' ')
        n_layer=$(echo $cfg | cut -f 6 -d ' ')
        max_step=$(echo $cfg | cut -f 7 -d ' ')
        n_head=$(echo $cfg | cut -f 8 -d ' ')
        eps=$(echo $cfg | cut -f 9 -d ' ')
        batch_size=$(echo $cfg | cut -f 10 -d ' ')
        block_length=$(echo $cfg | cut -f 11 -d ' ')


        CUDA_VISIBLE_DEVICES=0 python train_baseline.py \
            --cuda \
            --data ../penn/ \
            --dataset ptb \
            --n_layer ${n_layer} \
            --seed ${seed} \
            --d_model 256 \
            --n_head ${n_head} \
            --d_head 40 \
            --d_inner 2100 \
            --dropout 0.3 \
            --work_dir ${ckpt_src_path} \
            --dropatt 0.0 \
            --optim ${optim} \
            --lr ${lr} \
            --max_step ${max_step} \
            --tgt_len 32 \
            --mem_len 0 \
            --eval_tgt_len 32 \
            --batch_size ${batch_size} \
            --eps ${eps} \
            --block_length ${block_length} \
            --gpu0_bsz 1 \
            --max_epoch ${max_epoch} \
            --log-interval ${esd_interval} \
            --debug \
            
    done
