# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import os

from cv2 import inRange
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import ESDs.config as cf
import torch.optim as optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.yolo.engine.cal_scheduler import cosine_decay
from ESDs.train_utils_tbr import get_layer_temps, net_esd_estimator

from ESDs.sgdsnr import SGDSNR

class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to last checkpoint.
        best (Path): Path to best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            raise NotImplementedError("save_dir loss")
            #self.save_dir = Path(
            #    increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
       
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.best_ap = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]
        #self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
                self.args.rect = False
            # Command
            cmd, file = generate_ddp_command(2, self)
            try:
                LOGGER.info(f'Running DDP command {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        LOGGER.info(f'DDP settings: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
        dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo',
                                timeout=timedelta(seconds=3600),
                                rank=RANK,
                                world_size=world_size)

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        self.model.eval()
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        
        ###########################ESD analysis###########################
        ##################################################################
        print("----------------Start ESD analysis--------------")
        dir = self.save_dir / 'stats'
        if not os.path.exists(dir):
            os.makedirs(dir)

        filtered_layers = []
        metrics = net_esd_estimator(self.model, 
                  EVALS_THRESH = 0.00001,
                  bins = 100,
                  fix_fingers=self.args.fix_fingers,
                  xmin_pos=self.args.xmin_pos, 
                  filter_zeros = self.args.filter_zeros=='True')
        
        pd.DataFrame(metrics).to_csv(os.path.join(self.save_dir, 'stats',  f"metrics.csv")) 
        
        # summary and submit to wandb
        metric_summary = {}
        for key in metrics:
            if key != 'eigs' and key != 'longname':
                metric_summary[key] = np.mean(metrics[key])

        #######################  Filter out layers who has little amount of eigenvalues ##########################
        layer_with_few_eigs = []
        for i, name in enumerate(metrics['longname']):
            if len(metrics['eigs'][i]) <= self.args.tb_eig_filter:
                print(f"layer [{name}] has {len(metrics['eigs'][i])} eigenvalues, less than or equal to {self.args.tb_eig_filter}, remove it")
                layer_with_few_eigs.append(name)


        layer_stats=pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})      
        layer_stats_origin = layer_stats.copy()
        
        pd.DataFrame(layer_with_few_eigs).to_csv(os.path.join(self.save_dir, 'stats',  f"removed layers.csv")) 
        layer_stats_origin.to_csv(os.path.join(self.save_dir, 'stats',  f"origin_layer_stats_epoch_start.csv"))
        np.save(os.path.join(self.save_dir, 'stats', 'esd_epoch_{0}.npy'), metrics)


        ###################End  ESD analysis############################
        ##################################################################
        ######################  TBR scheduling ##########################
        ##################################################################
        if self.args.temp_balance_lr != 'None':
            print("--------------Enable temp balance --------------")
            
            if self.args.remove_first_layer == 'True':
                print("remove first layer of alpha<---------------------")
                layer_stats = layer_stats.drop(labels=0, axis=0)
                # index must be reset otherwise may delete the wrong row 
                layer_stats.index = list(range(len(layer_stats[self.args.metric])))
            if self.args.remove_last_layer == 'True':
                print("remove last layer of alpha<---------------------")
                layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0)
                # index must be reset otherwise may delete the wrong row 
                layer_stats.index = list(range(len(layer_stats[self.args.metric])))

            ####remove with the few eig values
            drop_layers = layer_stats['longname'].isin(layer_with_few_eigs)
            layer_stats = layer_stats[~drop_layers]
            
            metric_scores = np.array(layer_stats[self.args.metric])
            #args, temp_balance, n_alphas, epoch_val
            scheduled_lr = get_layer_temps(self.args, temp_balance=self.args.temp_balance_lr, n_alphas=metric_scores, epoch_val=self.args.lr0)
            layer_stats['scheduled_lr'] = scheduled_lr

            # these params should be tuned
            layer_name_to_tune = list(layer_stats['longname'])
            all_params = []
            params_to_tune_ids = []

            # these params should be tuned
            for name, module in self.model.named_modules():
                # these are the conv layers
                if name in layer_name_to_tune:
                    params_to_tune_ids += list(map(id, module.parameters()))
                    scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                    all_params.append({'params': module.parameters(), 'lr': scheduled_lr})
                # decide should we tune the batch norm accordingly,  is this layer batchnorm and does its corresponding conv in layer_name_to_tune
                elif self.args.batchnorm == 'True' \
                        and isinstance(module, nn.BatchNorm2d) \
                            and name.replace('bn', 'conv') in layer_name_to_tune:
                    params_to_tune_ids += list(map(id, module.parameters()))
                    scheduled_lr = layer_stats[layer_stats['longname'] == name.replace('bn', 'conv')]['scheduled_lr'].item()
                    all_params.append({'params': module.parameters(), 'lr': scheduled_lr})
                # another way is to add a else here and append params with self.args.lr0

            # those params are untuned
            untuned_params = filter(lambda p: id(p) not in params_to_tune_ids, self.model.parameters())
            all_params.append({'params': untuned_params, 'lr': self.args.lr0}) 

            # create optimizer
            if self.args.optim_type == 'SGDP':
                print(f"---->>>> Initialze the SGDP with lr {self.args.lr0}  {weight_decay}")
                optimizer = SGDP(all_params, 
                            lr=self.args.lr0,  
                            momentum=0.9, 
                            weight_decay=weight_decay)
            elif self.args.optim_type == 'SNR':
                optimizer = SGDSNR(all_params, 
                                    momentum=0.9, 
                                    weight_decay=weight_decay, 
                                    spectrum_regularization=self.args.sg,
                                    stage_epoch=self.args.stage_epoch,
                                    epoch=1)
            elif self.args.optim_type == 'SGD':
                optimizer = optim.SGD(all_params, 
                                    lr=self.args.lr0,  
                                    momentum=0.9, 
                                    weight_decay=weight_decay) 
            elif self.args.optim_type == 'Adam':
                optimizer = optim.Adam(all_params, 
                                    lr=self.args.lr0,  
                                    weight_decay=weight_decay) 
            elif self.args.optim_type == 'AdamW':
                optimizer = optim.AdamW(all_params, 
                                    lr=self.args.lr0,  
                                    weight_decay=weight_decay) 
            else:
                raise NotImplementedError
        else:
            print("-------------> Disable temp balance")
            if self.args.optim_type == 'SNR':
                optimizer = SGDSNR(self.model.parameters(), 
                                    momentum=0.9, 
                                    lr=self.args.lr0,  
                                    weight_decay=weight_decay, 
                                    spectrum_regularization=self.args.sg,
                                    stage_epoch=self.args.stage_epoch,
                                    epoch=1)
            elif self.args.optim_type == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), 
                                    lr=self.args.lr0,  
                                    momentum=0.9, 
                                    weight_decay=weight_decay) 
            elif self.args.optim_type == 'Adam':
                optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.lr0,  
                                    weight_decay=weight_decay) 
            elif self.args.optim_type == 'AdamW':
                optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.args.lr0,  
                                    weight_decay=weight_decay) 
            else:
                raise NotImplementedError

            
        # save scheduled learning rate 
        layer_stats.to_csv(os.path.join(self.save_dir, 'stats', f"layer_stats_with_lr_epoch_{0}.csv"))

        #########################################################################

        if self.args.lr_sche == 'step':
            self.lr_schedule = cf.stepwise_decay
        elif self.args.lr_sche == 'cosine':
            self.lr_schedule = cf.cosine_decay
        elif self.args.lr_sche == 'warmup_cosine':
            self.lr_schedule = cf.warmup_cosine_decay
        else:
            raise NotImplementedError

        elapsed_time = 0

       
    

        ##############################################################################################
        # Optimizer YOLO
        self.optimizer = optimizer

        self.lf = lambda x: 1
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        #Early stopping
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
       
        
        # Dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)

        ####training statistics
        training_stats = \
        {'precision': [],
        'recall': [],
        'mAP50': [],
        'mAP': [],
        'current_lr':[],
        'schedule_next_lr':[]
        }

        self._setup_train(world_size)
        self.elapsed_time = 0.0
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        #nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        
        self.untuned_lr = self.args.lr0
        is_current_best=False
        ##########################################
        last_model = self.model
    
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')

            # consider use another (maybe bigger) minimum learning rate in tbr
            if self.args.stage_epoch > 0 and epoch >= self.args.stage_epoch:
                print("------> Enter the second stage!!!!!!!!!!")
                self.args.lr0_min_ratio = self.args.lr0_min_ratio_stage2
            else:
                pass
            
            # this is current LR
            current_lr = self.untuned_lr
            print(f"##############Epoch {epoch+1}  current LR: {current_lr:.8f}################")

            ####################start training#####################
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                '''########################warm up code#######################
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch) ])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                '''

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log            
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')
            ########epoch end#############

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')
                 # save in interval
                state = {
                    
                        'net': self.model.state_dict(),
                        'precision': self.metrics['metrics/precision(B)'],
                        'recall': self.metrics['metrics/recall(B)'],
                        'mAP50': self.metrics['metrics/mAP50(B)'],
                        'mAP': self.metrics['metrics/mAP50-95(B)'],
                        'epoch':epoch
                    }
                #torch.save(state, os.path.join(self.save_dir, 'stats', f'epoch_{epoch}.ckpt'))
                # save best
                if self.best_fitness == self.fitness:
                    print('| Saving Best model')
                    state = {
                        'net': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'precision': self.metrics['metrics/precision(B)'],
                        'recall': self.metrics['metrics/recall(B)'],
                        'mAP50': self.metrics['metrics/mAP50(B)'],
                        'mAP': self.metrics['metrics/mAP50-95(B)'],
                        'epoch':epoch
                    }
                #    torch.save(state, os.path.join(self.save_dir, 'stats', f'epoch_best.ckpt'))

                
            #######################ESD analysis###############################
            ##################################################################
            self.model.eval()

            if self.epoch == 1 or self.epoch % self.args.ww_interval == 0:
                print("------------ Start ESD analysis -----------")
                a=datetime.now() 
                model_input = self.model


                metrics = net_esd_estimator(model_input, 
                            EVALS_THRESH = 0.00001,
                            bins = 100,
                            fix_fingers=self.args.fix_fingers,
                            xmin_pos=self.args.xmin_pos,
                            filter_zeros=self.args.filter_zeros=='True')
                
                metric_summary = {}
                for key in metrics:
                    if key != 'eigs' and key != 'longname':
                        metric_summary[key] = np.mean(metrics[key])

                
                #######################  Filter out layers who has little amount of eigenvalues ##########################
                layer_with_few_eigs = []
                for i, name in enumerate(metrics['longname']):
                    if len(metrics['eigs'][i]) <= self.args.tb_eig_filter:
                        print(f"layer [{name}] has {len(metrics['eigs'][i])} eigenvalues, less than or equal to {self.args.tb_eig_filter}, remove it")
                        layer_with_few_eigs.append(name)


                layer_stats=pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})      
                
                # save metrics to disk and ESD
                layer_stats_origin = layer_stats.copy()
                layer_stats_origin.to_csv(os.path.join(self.save_dir, 'stats',  f"origin_layer_stats_epoch_{epoch}.csv"))
                np.save(os.path.join(self.save_dir, 'stats', f'esd_epoch_{epoch}.npy'), metrics)
                if self.best_fitness == self.fitness:
                    np.save(os.path.join(self.save_dir, 'stats',f'esd_best.npy'), metrics)

                b=datetime.now() 
                print('seconds:', (b-a).seconds)
                ###################End  ESD analysis#############
            else:
                metric_summary = {}

            ##################################################################
            # Reschedule the learning rate
            self.untuned_lr = self.lr_schedule(self.args.lr0, epoch=self.epoch, total_epoch=self.epochs, warmup_epochs=self.args.warmup_epochs)
            print(f"------------>Rescheduled decayed LR: {self.untuned_lr:.8f}<--------------------")

            if self.args.temp_balance_lr != 'None':
                ######################  TBR scheduling ##########################
                ##################################################################

                print("---------- Schedule by Temp Balance---------------")
                assert len(metric_summary) > 0, "in TBR, every epoch should has an updated metric summary"
                if self.args.remove_first_layer == 'True':
                    print('remove first layer <--------------------')
                    layer_stats = layer_stats.drop(labels=0, axis=0)
                    # index must be reset otherwise next may delete the wrong row 
                    layer_stats.index = list(range(len(layer_stats[self.args.metric])))
                if self.args.remove_last_layer == 'True':
                    print('remove last layer <--------------------')
                    layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0)
                    # index must be reset otherwise may delete the wrong row 
                    layer_stats.index = list(range(len(layer_stats[self.args.metric])))
                
                ####remove with the few eig values
                drop_layers = layer_stats['longname'].isin(layer_with_few_eigs)
                layer_stats = layer_stats[~drop_layers]

                metric_scores = np.array(layer_stats[self.args.metric])
                scheduled_lr = get_layer_temps(self.args, self.args.temp_balance_lr, metric_scores, self.untuned_lr)
                layer_stats['scheduled_lr'] = scheduled_lr
                layer_name_to_tune = list(layer_stats['longname'])
                all_params_lr = []
                params_to_tune_ids = []
                c = 0
                
                #####check the few eig values layers were removed
                for name, module in self.model.named_modules():
                    if name in layer_name_to_tune:
                        assert name not in layer_with_few_eigs


                for name, module in self.model.named_modules():
                    if name in layer_name_to_tune:
                        params_to_tune_ids += list(map(id, module.parameters()))
                        scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                        all_params_lr.append(scheduled_lr)
                        c = c + 1
                    elif self.args.batchnorm == 'True' \
                        and isinstance(module, nn.BatchNorm2d) \
                            and name.replace('bn', 'conv') in layer_name_to_tune:
                        params_to_tune_ids += list(map(id, module.parameters()))
                        scheduled_lr = layer_stats[layer_stats['longname'] == name.replace('bn', 'conv')]['scheduled_lr'].item()
                        all_params_lr.append(scheduled_lr)
                        c = c + 1

                layer_stats.to_csv(os.path.join(self.save_dir, 'stats', f"layer_stats_with_lr_epoch_{self.epoch}.csv"))
                if self.best_fitness == self.fitness:
                    layer_stats.to_csv(os.path.join(self.save_dir, 'stats', f"layer_stats_with_lr_epoch_best.csv"))
                for index, param_group in enumerate(self.optimizer.param_groups):
                    #param_group['epoch'] = param_group['epoch'] + 1
                    if index <= c - 1:
                        param_group['lr'] = all_params_lr[index]
                    else:
                        param_group['lr'] = self.untuned_lr
            ##################################################################
            ##################################################################
            else:
                print("------------>  Schedule by default")
                for param_group in self.optimizer.param_groups:
                    #param_group['epoch'] = param_group['epoch'] + 1
                    param_group['lr'] = self.untuned_lr

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.elapsed_time += self.epoch_time
            print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(self.elapsed_time)))
            print('--------------------> <-----------------')
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            
            print("################ END Epoch#############")
        #########epoch circle end#############
        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            
            float_array = [epoch - self.start_epoch + 1, (time.time() - self.train_time_start)]
            df = pd.DataFrame({"Float Values": float_array})
            csv_file_path = "end_time.csv"
            df.to_csv(os.path.join(self.save_dir, 'stats', csv_file_path))

            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        """Save model checkpoints based on various conditions."""
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}


        # Save last, best and delete
        #torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt


    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build dataset"""
        raise NotImplementedError('build_dataset function not implemented in trainer')

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        raise NotImplementedError('criterion function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {'data': data, 'timestamp': time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args['data']).exists():
                    ckpt_args['data'] = self.args.data

                self.args = get_cfg(ckpt_args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
        return optimizer


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        a = m(im, device=device, verbose=False)[0].boxes.data  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.data  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. Setting 'amp=True'.")
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True
