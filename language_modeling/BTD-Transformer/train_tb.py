# coding: utf-8
import os
import time
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import random
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from models import *
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from esd_utils import net_esd_estimator, get_layer_temps


# set manual seed
def set_seed(seed=42):
    print(f"=====> Set the random seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data',                     type=str,           default='data/ptb',           help='location of the data corpus')
parser.add_argument('--dataset',                  type=str,           default='ptb',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'ptb'],                            help='dataset name')
parser.add_argument('--model',                    type=str,           default='tensorized',         help='transformer model used')
parser.add_argument('--n_layer',                  type=int,           default=12,                   help='number of total layers')
parser.add_argument('--n_head',                   type=int,           default=8,                    help='number of heads')
parser.add_argument('--d_head',                   type=int,           default=40,                   help='head dimension')
parser.add_argument('--d_embed',                  type=int,           default=-1,                   help='embedding dimension')
parser.add_argument('--d_model',                  type=int,           default=512,                  help='model dimension')
parser.add_argument('--d_inner',                  type=int,           default=2100,                 help='inner dimension in FF')
parser.add_argument('--dropout',                  type=float,         default=0.1,                  help='global dropout rate')
parser.add_argument('--dropatt',                  type=float,         default=0.0,                  help='attention probability dropout rate')
parser.add_argument('--init',                     type=str,           default='normal',              help='parameter initializer to use.')
parser.add_argument('--emb_init',                 type=str,           default='normal',             help='parameter initializer to use.')
parser.add_argument('--init_range',               type=float,         default=0.1,                  help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range',           type=float,         default=0.01,                 help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std',                 type=float,         default=0.02,                 help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std',            type=float,         default=0.01,                 help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim',                    type=str,           default='adam', 
                    choices=['adam', 'sgd'],   help='optimizer to use.')
parser.add_argument('--lr',                       type=float,         default=0.00025,              help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom',                      type=float,         default=0.0,                  help='momentum for sgd')
parser.add_argument('--scheduler',                type=str,           default='cosine', 
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],                         help='lr scheduler to use.')
parser.add_argument('--decay_rate',               type=float,         default=0.5,                  help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min',                   type=float,         default=0.0,                  help='minimum learning rate during annealing')
parser.add_argument('--clip',                     type=float,         default=0.25,                 help='gradient clipping')
parser.add_argument('--clip_nonemb',              action='store_true',                              help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step',                 type=int,           default=200000,               help='upper step limit')
parser.add_argument('--max_epoch',                type=int,           default=100,                  help='upper epoch limit')
parser.add_argument('--batch_size',               type=int,           default=60,                   help='batch size')
parser.add_argument('--batch_chunk',              type=int,           default=1,                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len',                  type=int,           default=32,                   help='number of tokens to predict')
parser.add_argument('--eval_tgt_len',             type=int,           default=32,                   help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len',                  type=int,           default=0,                    help='length of the extended context')
parser.add_argument('--mem_len',                  type=int,           default=32,                   help='length of the retained previous heads')
parser.add_argument('--not_tied',                 action='store_true',                              help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed',                     type=int,           default=1111,                 help='random seed')
parser.add_argument('--cuda',                     action='store_true',                              help='use CUDA')
parser.add_argument('--div_val',                  type=int,           default=1,                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm',                action='store_true',                              help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen',                   action='store_true',                              help='use variable length')
parser.add_argument('--multi_gpu',                action='store_true',                              help='use multiple GPU')
parser.add_argument('--log-interval',             type=int,           default=200,                  help='report interval')
parser.add_argument('--esd-interval',             type=int,           default=200,                  help='esd interval')
parser.add_argument('--eval-interval',            type=int,           default=1000,                 help='evaluation interval')
parser.add_argument('--work_dir',                 default='LM-TFM',   type=str,                     help='experiment directory.')
parser.add_argument('--restart',                  action='store_true',                              help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir',              type=str,           default='',                   help='restart dir')
parser.add_argument('--same_length',              action='store_true',                              help='use the same attn length for all tokens')
parser.add_argument('--attn_type',                type=int,           default=0,                    help='attention type. 0 for ours, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len',                type=int,           default=-1,                   help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min',                  type=float,         default=0.0,                  help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz',                 type=int,           default=4,                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps',           type=int,           default=-1,                   help='max eval steps')
parser.add_argument('--sample_softmax',           type=int,           default=-1,                   help='number of samples in sampled softmax')
parser.add_argument('--patience',                 type=int,           default=0,                    help='patience')
parser.add_argument('--finetune_v2',              action='store_true',                              help='finetune v2')
parser.add_argument('--finetune_v3',              action='store_true',                              help='finetune v3')
parser.add_argument('--fp16',                     action='store_true',                              help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale',        type=float,         default=1,                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale',       action='store_true',                              help='Use dynamic loss scaling.  If supplied, this argument supersedes --static-loss-scale.')

# esd related parameters
parser.add_argument('--pl-fitting',              type=str,           default='xmin_mid',           help="xmin_peak")
parser.add_argument('--filter-zeros',             type=str,           default='False')
parser.add_argument('--remove-last-layer',        type=str,           default='True',               help='if remove the last layer')
parser.add_argument('--remove-first-layer',       type=str,           default='True',               help='if remove the last layer')
parser.add_argument('--remove-few-eigs-layer',    type=str,           default='True',               help='if remove layers with few eigenvalues')
parser.add_argument('--metric',                   type=str,           default='alpha',              help='ww metric')
parser.add_argument('--assign-func',              type=str,           default='tb_linear_map',      help='use tempbalance for learning rate')
parser.add_argument('--layernorm',                type=str,           default='False')
parser.add_argument('--lr-min-ratio',             type=float,         default=0.7)
parser.add_argument('--lr-slope',                 type=float,         default=0.6)
parser.add_argument('--xmin-pos',                 type=float,         default=2,                    help='xmin_index = size of eigs // xmin_pos')
parser.add_argument('--eigs-thresh',              type=int,           default=50)
parser.add_argument('--tb-update',                type=str,           default='spike',               help='how lrs are updated during the interval')

# adabelief related parameters
parser.add_argument('--wdecay',                   type=float,         default=1.2e-6,               help='weight decay')
parser.add_argument('--eps',                      type=float,         default=1e-8)
parser.add_argument('--beta1',                    type=float,         default=0.9,                  help='beta1 value')
parser.add_argument('--beta2',                    type=float,         default=0.999,                help='bets2 value')
parser.add_argument('--block_length',             type=int,           default=4,                    help='block_length')

# special argument to remove the out linear layer
parser.add_argument('--metric-scores', nargs='+', type=float,         default=None)
parser.add_argument('--layer_stats',              type=str,           default=None)

args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

# modify work directory
args.work_dir = '{}/{}/{}/{}-{}'.format(args.work_dir, args.model, f'tb_{args.tb_update}_update', args.dataset, args.optim)
if args.optim.lower() == 'adam':
    args.work_dir = '{}/bs{}'.format(args.work_dir, args.batch_size)
if args.remove_last_layer == 'True':
    args.work_dir = '{}-{}'.format(args.work_dir, 'remove_last')
if args.remove_few_eigs_layer == 'True':
    args.work_dir = '{}-{}'.format(args.work_dir, 'eigs_thresh_{}'.format(args.eigs_thresh))

    
args.work_dir = os.path.join(args.work_dir, 
                            f'tensor_transformer_{args.n_layer}layer', 
                            f'head_{args.n_head}', \
                            f"max_step{args.max_step}_max_epoch{args.max_epoch}_log_interval{args.log_interval}_esd_interval{args.esd_interval}", \
                            f"{args.metric}_{args.pl_fitting}_min{args.lr_min_ratio}_slope{args.lr_slope}_xmin_pos{args.xmin_pos}_assign_{args.assign_func}", \
                            f"seed_{args.seed}_lr_{args.lr}")

logging = create_exp_dir(args.work_dir, debug=False) #scripts_to_save=['train_zs.py', 'mem_transformer_zs.py'], 

# Set the random seed manually for reproducibility.
set_seed(args.seed)
# torch.cuda.set_device(1)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)


device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]

# initialize checkpoints and debug stats
training_stats = {
    'train_loss': [],
    'train_ppl': [],
    'val_loss': [],
    'val_ppl': [],
    'test_loss': [],
    'test_ppl': [],
    'step': [],
    'lr': [],
    'epoch_duration': []
}



###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    # init
    elif classname.find('MultiHeadAttn') != -1:
        if hasattr(m, 'core_value'):
            for i in range(m.core_nums):
                nn.init.normal_(m.core_value[i], 0.0, args.proj_init_std)

    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    if args.model == 'tensorized':
        model = TensorizedTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
                                args.d_head, args.d_inner, args.dropout, args.dropatt,
                                tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                                tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                                ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                                same_length=args.same_length, attn_type=args.attn_type,
                                clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(weights_init)
        model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing
    else:
        raise NotImplementedError
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
self_attention_param = 0
for p in model.layers:
    for a in p.dec_attn.parameters():
        self_attention_param += a.nelement()

args.self_attention_param = self_attention_param

if args.fp16:
    model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    logging('Training on Single GPU......')
    para_model = model.to(device)

print(model)

# initialize untuned lr
untuned_lr = args.lr

# create mapping from layernorm to linear layer
logging("#####Mapping layer norm to Linear#######")
longname_lst = []
type_lst = []
ln_to_linear = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        longname_lst.append(name)
        type_lst.append('nn.Linear')
    if isinstance(module, nn.LayerNorm):
        if type_lst[-1] == 'nn.Linear':
            ln_to_linear[name] = longname_lst[-1]
        longname_lst.append(name)
        type_lst.append('nn.LayerNorm')
for key in ln_to_linear:
    logging(f"{key} -> {ln_to_linear[key]}")

#######################ESD analysis###############################
##################################################################
logging("####################Start ESD analysis###################")
if not os.path.exists(os.path.join(args.work_dir, 'stats')):
    os.makedirs(os.path.join(args.work_dir, 'stats'))
metrics = net_esd_estimator(model, 
                  EVALS_THRESH = 0.00001,
                  bins = 100,
                  pl_fitting=args.pl_fitting,
                  xmin_pos=args.xmin_pos, 
                  filter_zeros = args.filter_zeros=='True')

args.layer_stats=pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
np.save(os.path.join(args.work_dir, 'stats', 'esd_epoch_0.npy'), metrics)

######################  TBR scheduling ##########################
##################################################################
logging("################## Enable temp balance ##############")
if args.remove_last_layer == 'True':
    logging("remove last layer of alpha<---------------------")
    args.layer_stats = args.layer_stats.drop(labels=len(args.layer_stats) - 1, axis=0)
    # index must be reset otherwise may delete the wrong row 
    args.layer_stats.index = list(range(len(args.layer_stats[args.metric])))
if args.remove_few_eigs_layer == 'True':
    assert args.eigs_thresh > 0, "Must set a valid threshold to filter the eigenvaues"
    layer_with_few_eigs = []
    for i, name in enumerate(metrics['longname']):
        if len(metrics['eigs'][i]) < args.eigs_thresh:
            logging(f"layer [{name}] has {len(metrics['eigs'][i])} eigenvalues, less than {args.eigs_thresh}, remove it")
            layer_with_few_eigs.append(name)
    logging(f"remove {len(layer_with_few_eigs)} layers with few eigs<---------------------")
    drop_layers = args.layer_stats['longname'].isin(layer_with_few_eigs)
    args.layer_stats = args.layer_stats[~drop_layers]


args.metric_scores = np.array(args.layer_stats[args.metric])
#args, n_alphas, epoch_val
scheduled_lr_lst = get_layer_temps(args, args.assign_func, args.metric_scores, args.lr)
args.layer_stats['scheduled_lr'] = scheduled_lr_lst

# these params should be tuned
layer_name_to_tune = list(args.layer_stats['longname'])
all_params = []
all_params_lr = []
params_to_tune_ids = []
linear_count = 0
norm_count = 0
all_count = 0

# these params should be tuned
for name, module in model.named_modules():
    # these are the conv layers analyzed by 
    if name in layer_name_to_tune:
        assert name not in layer_with_few_eigs
        params_to_tune_ids += list(map(id, module.parameters()))
        scheduled_lr = args.layer_stats[args.layer_stats['longname'] == name]['scheduled_lr'].item()
        all_params_lr.append(scheduled_lr)
        all_params.append({'params': module.parameters(), 'lr': args.lr})
        linear_count += 1
        all_count += 1
    # decide should we tune the batch norm accordingly,  is this layer batchnorm and does its corresponding conv in layer_name_to_tune
    elif args.layernorm == 'True' \
            and isinstance(module, nn.LayerNorm) \
                and name in ln_to_linear \
                    and ln_to_linear[name] in layer_name_to_tune:

        #logging(f"Initial Tuning LayerNorm: {name} -------> {ln_to_linear[name]} ")
        params_to_tune_ids += list(map(id, module.parameters()))
        scheduled_lr = args.layer_stats[args.layer_stats['longname'] == ln_to_linear[name]]['scheduled_lr'].item()
        all_params_lr.append(scheduled_lr)
        all_params.append({'params': module.parameters(), 'lr': args.lr})
        norm_count += 1
        all_count += 1
    # another way is to add a else here and append params with args.lr
    elif name in layer_with_few_eigs:
        print(list(map(id, module.parameters())))

# those params are untuned
logging(f"Total number of params to tune : {all_count},  linear: {linear_count}  norm: {norm_count}")
untuned_params = filter(lambda p: id(p) not in params_to_tune_ids, model.parameters())
all_params.append({'params': untuned_params, 'lr': args.lr}) 


#### optimizer
if args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_params, lr=args.lr)
else:
    raise NotImplementedError



logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))
logging('#self attention params = {}'.format(args.self_attention_param))


def cosine_decay(init, epoch, total_epoch):
    epoch = min(epoch, total_epoch)
    cosine_decay = 0.5 * (1 + math.cos(np.pi * epoch / total_epoch))
    
    return init * cosine_decay

###############################################################################
# Training and Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    logging(f'total_loss: {total_loss}, total_len: {total_len}')

    return total_loss / total_len

def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time, untuned_lr, pbar, epoch, all_count
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()


        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # record learning rate
        prev_lr = []
        for param_group in optimizer.param_groups:
            prev_lr.append(param_group['lr'])

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1

        # get untuned base learning rate with cosine annealing
        untuned_lr = cosine_decay(args.lr, train_step, args.max_step)

        ##################################################################
        # Reschedule the learning rate
        if train_step % args.esd_interval == 0:
            logging("################ Start ESD analysis#############")
            esd_start_time = time.time()
            metrics = net_esd_estimator(model, 
                    EVALS_THRESH = 0.00001,
                    bins = 100,
                    pl_fitting=args.pl_fitting,
                    xmin_pos=args.xmin_pos,
                    filter_zeros=args.filter_zeros=='True')
            
            metric_summary = {}
            for key in metrics:
                if key != 'eigs' and key != 'longname':
                    metric_summary[key] = np.mean(metrics[key])

            args.layer_stats= pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})

            esd_estimated_time = time.time() - esd_start_time
            logging(f"-----> ESD estimation time: {esd_estimated_time:.3f}")

            logging("############### Schedule by Temp Balance###############")
            assert len(metric_summary) > 0, "in TBR, every epoch should has an updated metric summary"
            if args.remove_last_layer == 'True':
                print('remove last layer <--------------------')
                args.layer_stats = args.layer_stats.drop(labels=len(args.layer_stats) - 1, axis=0)
                # index must be reset otherwise may delete the wrong row 
                args.layer_stats.index = list(range(len(args.layer_stats[args.metric])))
            if args.remove_few_eigs_layer == 'True':
                print(f"remove {len(layer_with_few_eigs)} layers with few eigs<---------------------")
                drop_layers = args.layer_stats['longname'].isin(layer_with_few_eigs)
                args.layer_stats = args.layer_stats[~drop_layers]
            
            args.metric_scores = np.array(args.layer_stats[args.metric])
            
            print(f"---------------------->>> args.metric_scores has been updated <<<---------------------")

            scheduled_lr_lst = get_layer_temps(args, args.assign_func, args.metric_scores, untuned_lr)
            args.layer_stats['scheduled_lr'] = scheduled_lr_lst
        
            layer_name_to_tune = list(args.layer_stats['longname'])
            all_params_lr = []

            linear_count = 0
            norm_count = 0
            all_count = 0
            for name, module in model.named_modules():
                if name in layer_name_to_tune:
                    assert name not in layer_with_few_eigs
                    scheduled_lr = args.layer_stats[args.layer_stats['longname'] == name]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    linear_count += 1
                    all_count += 1
                elif args.layernorm == 'True' \
                        and isinstance(module, nn.LayerNorm) \
                            and name in ln_to_linear \
                                and ln_to_linear[name] in layer_name_to_tune:

                    scheduled_lr = args.layer_stats[args.layer_stats['longname'] == ln_to_linear[name]]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    norm_count += 1
                    all_count += 1

            for index, param_group in enumerate(optimizer.param_groups):
                if index <= all_count - 1:
                    param_group['lr'] = all_params_lr[index]
                else:
                    param_group['lr'] = untuned_lr
 

        else:  
            if args.tb_update == 'stage': # 
                for index, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = prev_lr[index]
            else:
                raise NotImplementedError

        if train_step % args.log_interval == 0:
            ##################### log training stats ###########################
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                        '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch + 1, untuned_lr,
                                    elapsed * 1000 / args.log_interval, cur_loss)
            training_stats['train_loss'].append(cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                training_stats['train_ppl'].append(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

            ###################### test on val set #######################
            val_loss = evaluate(va_iter)
            test_loss = evaluate(te_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                        '| valid loss {:5.2f}'.format(
                train_step // args.log_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)
            
            training_stats['val_loss'].append(val_loss)
            training_stats['val_ppl'].append(math.exp(val_loss))
            training_stats['test_loss'].append(test_loss)
            training_stats['test_ppl'].append(math.exp(test_loss))
            training_stats['step'].append(train_step)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                PATH=os.path.join(args.work_dir, 'model.pt')
                torch.save({
                    'val_loss': val_loss,
                    'val_ppl': math.exp(val_loss),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
                # np.save(os.path.join(args.work_dir, f'esd_best.npy'), metrics)
                best_val_loss = val_loss
            
            eval_start_time = time.time()
            
            np.save(os.path.join(args.work_dir, "training_stats.npy"), training_stats)
                    
        pbar.update(1)
        if train_step == args.max_step:
            break

# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# test initialization
init_train_loss = evaluate(tr_iter)
init_val_loss = evaluate(va_iter)
init_test_loss = evaluate(te_iter)
logging('| Start of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        init_test_loss, math.exp(init_test_loss)))
training_stats['train_loss'].append(init_train_loss)
training_stats['val_loss'].append(init_test_loss)
training_stats['test_loss'].append(init_test_loss)
training_stats['train_ppl'].append(math.exp(init_train_loss))
training_stats['val_ppl'].append(math.exp(init_test_loss))
training_stats['test_ppl'].append(math.exp(init_test_loss))
training_stats['step'].append(0)

pbar = tqdm.tqdm(total=args.max_step)
total_start_time = time.time()
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()
        train()
        epoch_duration = time.time() - epoch_start_time
        training_stats['epoch_duration'].append(epoch_duration)
        if train_step == args.max_step or epoch == args.max_epoch:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')
total_duration= time.time() - total_start_time
training_stats['total_duration'] = total_duration

np.save(os.path.join(args.work_dir, "training_stats.npy"), training_stats)
# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    state_dict = torch.load(f)
    model.load_state_dict(state_dict['model_state_dict'])
para_model = model.to(device)

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)))
else:
    logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))
    training_stats['test_loss'].append(test_loss)
    training_stats['test_ppl'].append(math.exp(test_loss))
logging('=' * 100)
np.save(os.path.join(args.work_dir, "training_stats.npy"), training_stats)