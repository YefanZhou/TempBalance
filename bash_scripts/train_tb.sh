source ~/.bashrc
conda activate ww_train
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

SLURM_ARRAY_TASK_ID=$1
src_path=$2               #/home/eecs/yefan0726/ww_train_repos/TempBalance
ckpt_src_path=$3          #/data/yefan0726/checkpoints/tempbalance
data_path=$4              #/data/yefan0726/data/cv

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/bash_scripts/config/tb.txt)

netType=$(echo $cfg | cut -f 1 -d ' ')
width=$(echo $cfg | cut -f 2 -d ' ')
depth=$(echo $cfg | cut -f 3 -d ' ')
dataset=$(echo $cfg | cut -f 4 -d ' ')
num_epochs=$(echo $cfg | cut -f 5 -d ' ')
lr_sche=$(echo $cfg | cut -f 6 -d ' ')
seed_lst=$(echo $cfg | cut -f 7 -d ' ')
lr=$(echo $cfg | cut -f 8 -d ' ')
weight_decay=$(echo $cfg | cut -f 9 -d ' ')
pl_fitting=$(echo $cfg | cut -f 10 -d ' ')
remove_first=$(echo $cfg | cut -f 11 -d ' ')
remove_last=$(echo $cfg | cut -f 12 -d ' ')
metric=$(echo $cfg | cut -f 13 -d ' ')
assign_func=$(echo $cfg | cut -f 14 -d ' ')
batchnorm=$(echo $cfg | cut -f 15 -d ' ')
lr_min_ratio=$(echo $cfg | cut -f 16 -d ' ')
lr_max_ratio=$(echo $cfg | cut -f 17 -d ' ')
xmin_pos=$(echo $cfg | cut -f 18 -d ' ')
sg=$(echo $cfg | cut -f 19 -d ' ')
bn_type=$(echo $cfg | cut -f 20 -d ' ')
use_tb=$(echo $cfg | cut -f 21 -d ' ')
optim_type=$(echo $cfg | cut -f 22 -d ' ')



# saving folders of trained models 
base_path=${ckpt_src_path}/${dataset}/TB_${use_tb}/${netType}_${depth}_${width}
base_path=${base_path}/${metric}_plfitting_${pl_fitting}_xminpos${xmin_pos}_${assign_func}
base_path=${base_path}/${optim_type}_min${lr_min_ratio}_max${lr_max_ratio}_init${lr}_${lr_sche}
base_path=${base_path}/snr_sg${sg}/refirst${remove_first}_relast${remove_last}_bn${batchnorm}_type_${bn_type}

array=($(echo $seed_lst | tr ',' ' '))

# run five random seeds 
for seed in 43 37 13 51 71
    do 
        echo run experiment with seed $seed
        ckpt_folder=epochs${num_epochs}_wd${weight_decay}_seed${seed}
        mkdir -p ${base_path}/${ckpt_folder}

        python main_tb.py \
            --lr ${lr} \
            --net-type ${netType} \
            --depth ${depth} \
            --widen-factor ${width} \
            --num-epochs ${num_epochs} \
            --seed ${seed} \
            --dataset ${dataset} \
            --use-tb ${use_tb} \
            --optim-type ${optim_type} \
            --lr-sche ${lr_sche} \
            --weight-decay ${weight_decay} \
            --pl-fitting ${pl_fitting} \
            --remove-last-layer ${remove_last} \
            --remove-first-layer ${remove_first} \
            --esd-metric-for-tb ${metric} \
            --assign-func ${assign_func} \
            --batchnorm ${batchnorm} \
            --lr-min-ratio ${lr_min_ratio} \
            --lr-max-ratio ${lr_max_ratio} \
            --xmin-pos ${xmin_pos} \
            --batchnorm-type ${bn_type} \
            --sg ${sg} \
            --ckpt-path ${base_path}/${ckpt_folder} \
            --datadir ${data_path}
    done
