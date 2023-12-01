pwd
hostname
date
echo starting test job...
source ~/.bashrc
conda activate tb_yolo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1


root=$(pwd)
ckpt_root=$(pwd)/checkpoints

for SLURM_ARRAY_TASK_ID in {2..2..1}
    do 

        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${root}/scripts/txt_files/baseline.txt)
        netType=$(echo $cfg | cut -f 1 -d ' ')
        width=$(echo $cfg | cut -f 2 -d ' ')        #NO NEEDED
        depth=$(echo $cfg | cut -f 3 -d ' ')        #NO NEEDED 
        dataset=$(echo $cfg | cut -f 4 -d ' ')
        num_epochs=$(echo $cfg | cut -f 5 -d ' ')
        lr_sche=$(echo $cfg | cut -f 6 -d ' ')
        seed=$(echo $cfg | cut -f 7 -d ' ')
        lr=$(echo $cfg | cut -f 8 -d ' ')
        weight_decay=$(echo $cfg | cut -f 9 -d ' ')
        warmup_epochs=$(echo $cfg | cut -f 10 -d ' ')
        fix_fingers=$(echo $cfg | cut -f 11 -d ' ')
        remove_first=$(echo $cfg | cut -f 12 -d ' ')
        remove_last=$(echo $cfg | cut -f 13 -d ' ')
        metric=$(echo $cfg | cut -f 14 -d ' ')
        temp_balance_lr=$(echo $cfg | cut -f 15 -d ' ')
        batchnorm=$(echo $cfg | cut -f 16 -d ' ')
        lr_min_ratio=$(echo $cfg | cut -f 17 -d ' ')
        lr_slope=$(echo $cfg | cut -f 18 -d ' ')
        xmin_pos=$(echo $cfg | cut -f 19 -d ' ')
        sg=$(echo $cfg | cut -f 20 -d ' ')
        stage_epoch=$(echo $cfg | cut -f 21 -d ' ')
        lr_min_ratio2=$(echo $cfg | cut -f 22 -d ' ')
        optim_type=$(echo $cfg | cut -f 23 -d ' ')

        base_path=${ckpt_root}/${netType}/${metric}_fixf${fix_fingers}_xminpos${xmin_pos}_${temp_balance_lr}_optim${optim_type}
        base_path=${base_path}/min${lr_min_ratio}_slope${lr_slope}/sg${sg}_stage${stage_epoch}_lrmin${lr_min_ratio2}/refirst${remove_first}_relast${remove_last}_bn${batchnorm}
        

        save_path_test=${ckpt_root}/val/${netType}/${metric}_fixf${fix_fingers}_xminpos${xmin_pos}_${temp_balance_lr}_optim${optim_type}
        save_path_test=${save_path_test}/min${lr_min_ratio}_slope${lr_slope}/sg${sg}_stage${stage_epoch}_lrmin${lr_min_ratio2}/refirst${remove_first}_relast${remove_last}_bn${batchnorm}

        ckpt_folder=${netType}_${dataset}_${lr}_${lr_sche}_${num_epochs}_wd${weight_decay}_seed${seed}_warm${warmup_epochs}
        
        read_model_dir=${base_path}/${ckpt_folder}/weights/best.pt
        
        mkdir -p ${save_path_test}/${ckpt_folder}
        echo ................Start...........
        CUDA_VISIBLE_DEVICES=7 python YOLOv8/runval.py \
            --net-type ${netType} \
            --dataset ${dataset} \
            --ckpt-path ${save_path_test}/${ckpt_folder} \
            --read-model-dir ${read_model_dir} 
        echo ................Next Stage...........
        
    done