
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww_train_lm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

cd ${src_path}/BTD-Transformer/

for SLURM_ARRAY_TASK_ID in {2..6..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/BTD-Transformer/scripts/txt/ptb_tb.txt)

        lr=$(echo $cfg | cut -f 1 -d ' ')
        seed=$(echo $cfg | cut -f 2 -d ' ')
        pl_fitting=$(echo $cfg | cut -f 3 -d ' ')
        remove_first=$(echo $cfg | cut -f 4 -d ' ')
        remove_last=$(echo $cfg | cut -f 5 -d ' ')
        metric=$(echo $cfg | cut -f 6 -d ' ')
        assign_func=$(echo $cfg | cut -f 7 -d ' ')
        layernorm=$(echo $cfg | cut -f 8 -d ' ')
        lr_min_ratio=$(echo $cfg | cut -f 9 -d ' ')
        lr_slope=$(echo $cfg | cut -f 10 -d ' ')
        xmin_pos=$(echo $cfg | cut -f 11 -d ' ')
        max_epoch=$(echo $cfg | cut -f 12 -d ' ')
        esd_interval=$(echo $cfg | cut -f 13 -d ' ')
        optim=$(echo $cfg | cut -f 14 -d ' ')
        n_layer=$(echo $cfg | cut -f 15 -d ' ')
        max_step=$(echo $cfg | cut -f 16 -d ' ')
        n_head=$(echo $cfg | cut -f 17 -d ' ')
        batch_size=$(echo $cfg | cut -f 18 -d ' ')
        tb_update=$(echo $cfg | cut -f 19 -d ' ')

        CUDA_VISIBLE_DEVICES=0 python train_tb.py \
            --cuda \
            --data ../penn/ \
            --dataset ptb \
            --n_layer ${n_layer} \
            --seed ${seed} \
            --d_model 256 \
            --n_head 1 \
            --d_head 40 \
            --d_inner 2100 \
            --dropout 0.3 \
            --work_dir ${ckpt_src_path} \
            --dropatt 0.0 \
            --optim ${optim} \
            --lr ${lr} \
            --max_step 40000 \
            --tgt_len 32 \
            --mem_len 0 \
            --eval_tgt_len 32 \
            --batch_size ${batch_size} \
            --gpu0_bsz 1 \
            --pl-fitting ${pl_fitting} \
            --remove-first-layer ${remove_first} \
            --remove-last-layer ${remove_last} \
            --metric ${metric} \
            --assign-func ${assign_func} \
            --layernorm ${layernorm} \
            --lr-min-ratio ${lr_min_ratio} \
            --lr-slope ${lr_slope} \
            --xmin-pos ${xmin_pos} \
            --max_epoch ${max_epoch} \
            --esd-interval ${esd_interval} \
            --tb-update ${tb_update} 
    
    done