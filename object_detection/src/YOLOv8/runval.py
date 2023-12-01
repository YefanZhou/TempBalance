from sklearn import metrics
from ultralytics import YOLO
import argparse
import pandas as pd 
from pathlib import Path
parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--lr',             type=float,  default=0.01,                         help='learning_rate')
parser.add_argument('--net-type',       type=str,    default='yolov8n',                help='model')
parser.add_argument('--num-epochs',     type=int,    default=200,                          help='number of epochs')
parser.add_argument('--warmup-epochs',  type=int,    default=0) 
parser.add_argument('--dataset',        type=str,    default='VOC2007.yaml',                    help='dataset = [cifar10/cifar100]')
parser.add_argument('--lr-sche',        type=str,    default='cosine',                   choices=['step', 'cosine', 'warmup_cosine'])
parser.add_argument('--weight-decay',   type=float,  default=1e-4) # 5e-4
parser.add_argument('--wandb-tag',      type=str,    default='')
parser.add_argument('--wandb-on',       type=str,    default='False')
parser.add_argument('--print-tofile',   type=str,    default='False')
parser.add_argument('--ckpt-path',      type=str,    default='')


parser.add_argument('--batch-size',   type=int,          default=64) # 5e-4
parser.add_argument('--optim-type',     type=str,        default='SGDP',                        help='type of optimizer')
parser.add_argument('--resume',         type=str,        default='',                           help='resume from checkpoint')
parser.add_argument('--seed',           type=int,        default=42) 
parser.add_argument('--ww-interval',    type=int,        default=1)
parser.add_argument('--fix-fingers',     type=str,       default=None, help="xmin_peak")
parser.add_argument('--pl-package',     type=str,        default='powerlaw')

# temperature balance related 
parser.add_argument('--remove-last-layer',   type=str,    default='True',  help='if remove the last layer')
parser.add_argument('--remove-first-layer',  type=str,   default='True',  help='if remove the last layer')
parser.add_argument('--metric',                type=str,    default='alpha',  help='ww metric')  #tb_linear_map
parser.add_argument('--temp-balance-lr',       type=str,    default='None',       help='use tempbalance for learning rate')
parser.add_argument('--batchnorm',             type=str,    default='False')
parser.add_argument('--lr-min-ratio',          type=float,  default=0.8)
parser.add_argument('--lr-slope',           type=float,  default=0.7)
parser.add_argument('--xmin-pos',           type=float,  default=2, help='xmin_index = size of eigs // xmin_pos')
parser.add_argument('--lr-min-ratio-stage2',   type=float,  default=1)
# spectral regularization related
parser.add_argument('--sg',                 type=float, default=0.01, help='spectrum regularization')
parser.add_argument('--stage-epoch',        type=int, default=0,  help='stage_epoch')
parser.add_argument('--filter-zeros',  type=str,   default='False')
parser.add_argument('--read-model-dir',  type=str,   default='')
####There is Another parameter file named default.yaml located YOLOv8/ultralytics/yolo/cfg/default.yaml

class TB_PARAMS:
    args_tb = parser.parse_args()

# Load a model
#CUDA_VISIBLE_DEVICES=7 python yolov8_CAL_G/runtrain.py
model1 = YOLO(TB_PARAMS.args_tb.net_type + '.yaml')
model1 = YOLO(TB_PARAMS.args_tb.read_model_dir)
#test


print(TB_PARAMS.args_tb.ckpt_path)
metrics = model1.val(
            lr0 = TB_PARAMS.args_tb.lr,
            epochs = TB_PARAMS.args_tb.num_epochs,
            warmup_epochs = TB_PARAMS.args_tb.warmup_epochs,
            data = TB_PARAMS.args_tb.dataset,
            weight_decay = TB_PARAMS.args_tb.weight_decay,
            batch = TB_PARAMS.args_tb.batch_size,
            seed = TB_PARAMS.args_tb.seed,

            #extra
            save_dir = TB_PARAMS.args_tb.ckpt_path,
            lr_sche = TB_PARAMS.args_tb.lr_sche,
            ww_interval = TB_PARAMS.args_tb.ww_interval,
            optim_type = TB_PARAMS.args_tb.optim_type,

            #fitting
            fix_fingers =  TB_PARAMS.args_tb.fix_fingers,
            pl_package = TB_PARAMS.args_tb.pl_package,

            #TB related
            remove_first_layer = TB_PARAMS.args_tb.remove_first_layer,
            remove_last_layer = TB_PARAMS.args_tb.remove_last_layer,
            metric = TB_PARAMS.args_tb.metric,
            temp_balance_lr = TB_PARAMS.args_tb.temp_balance_lr,
            batchnorm = TB_PARAMS.args_tb.batchnorm,
            lr_min_ratio = TB_PARAMS.args_tb.lr_min_ratio,
            lr_slope = TB_PARAMS.args_tb.lr_slope,
            xmin_pos = TB_PARAMS.args_tb.xmin_pos,
            lr_min_ratio_stage2 = TB_PARAMS.args_tb.lr_min_ratio_stage2,

            #snr
            sg = TB_PARAMS.args_tb.sg,
            stage_epoch = TB_PARAMS.args_tb.stage_epoch
           )
#save to csv
metric_save = metrics.results_dict
df = pd.DataFrame(metric_save, index=[0])
df.to_csv(Path(TB_PARAMS.args_tb.ckpt_path) / "results.csv")
print("end")


