#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import gc
import os
import numpy as np
import time as Ti
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import Models.losses as losses
import Models.optimizer as optim
import Utils.checkpoint as cu
import Utils.distributed as du
import Mainframe.set_logging as set_logging
import Utils.metrics as metrics
import Utils.misc as misc
from Models import build_model
from Utils.meters import EpochTimer, TrainMeter, ValMeter

from typing import List
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField,IntField,TorchTensorField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder,NDArrayDecoder, RandomResizedCropRGBImageDecoder
from ffcv.pipeline.operation import Operation

import ipdb
from itertools import combinations
import random
import shutil

logger = set_logging.get_logger(__name__)    

def augmentation(cfg,frames, frames_idx):
    frames_idx_out = frames_idx
    
    # random flip
    if cfg.DATA.RANDOM_FLIP:
        if np.random.rand() > 0.5:
            frames_out = torch.flip(frames, dims=[3])             
        else:
            frames_out = frames
            
    # random crop
    crop_size = cfg.DATA.CROP_SIZE
    x_max_offset = int(math.ceil(cfg.DATA.RESIZE_SCALE_WIDTH - crop_size))
    y_max_offset = int(math.ceil(cfg.DATA.RESIZE_SCALE_HEIGHT - crop_size))
    x_offset = int(random.randint(0,x_max_offset))
    y_offset = int(random.randint(0,y_max_offset))
    frames_out = frames_out[:, :, int(y_offset) : int(y_offset) + crop_size, int(x_offset) : int(x_offset) + crop_size,:]    
    
    return frames_out, frames_idx_out

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    for cur_iter, (inputs, labels, frame_index,video_idx) in enumerate(
        train_loader
    ):  
        inputs = inputs.float()/255.0 
        inputs = (inputs - torch.tensor(cfg.DATA.MEAN))/(torch.tensor(cfg.DATA.STD))
        inputs,frame_index = augmentation(cfg,inputs,frame_index)
        inputs = inputs.permute(0, 4, 1, 2, 3) 
              
        #  Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
    
            labels = labels.squeeze(1).cuda(non_blocking=True)

        batch_size = (
            inputs[0].shape[0]
            if isinstance(inputs, list)
            else inputs.shape[0]
        )

        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            perform_backward = True
            optimizer.zero_grad()
              
            aux_loss,preds = model(inputs,labels)
            loss = 1.0*loss_fun(preds, labels)      
            loss += aux_loss

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if perform_backward:
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        grad_norm = optim.get_grad_norm_(model.parameters())
        
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()
        
        top1_err, top5_err = None, None

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, grad_norm, top1_err, top5_err = du.all_reduce(
                [loss.detach(), grad_norm, top1_err.detach(), top5_err.detach()]
            )

        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            grad_norm,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            None,
        )
    
        train_meter.iter_toc() 
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()

    del inputs
    torch.cuda.empty_cache()
    
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, last_epoch 
):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, frame_index,video_idx) in enumerate(val_loader):
        inputs = inputs.float()/255.0
        inputs = (inputs - torch.tensor(cfg.DATA.MEAN))/(torch.tensor(cfg.DATA.STD))
        inputs = inputs.permute(0, 4, 1, 2, 3) 
        
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
    
            labels = labels.squeeze(1).cuda(non_blocking=True)
            
        batch_size = (
            inputs[0].shape[0]
            if isinstance(inputs, list)
            else inputs.shape[0]
        )
        
        val_meter.data_toc()

        aux_loss,preds = model(inputs,labels)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    
    if last_epoch:
        result_string = (
            "Last Iter - Validation Top1 Acc: {}% Validation Top5 Acc: {}%"
            "".format(
                100.0-val_meter.stats["top1_err"],
                100.0-val_meter.stats["top5_err"],
            )
        )
        logger.info("{}".format(result_string))

    val_meter.reset()


def train(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    
    # Setup logging format.
    set_logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg,"train")

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    if du.is_master_proc(num_gpus=cfg.NUM_GPUS * cfg.NUM_SHARDS) and cfg.LOG_MODEL_INFO:
        for name, param in model.named_parameters():
            logger.info("{}:{}".format(name,param.requires_grad))
    
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    start_epoch = 0
    if cfg.TRAIN.FFCV.ENABLE or cfg.VAL.FFCV.ENABLE:
        video_pipeline:List[Operation] = [NDArrayDecoder(),ToTensor()]
        label_pipeline:List[Operation] = [IntDecoder(),ToTensor()]
        frame_idx_pipeline:List[Operation] = [NDArrayDecoder(),ToTensor()]
        video_idx_pipeline:List[Operation] = [IntDecoder(),ToTensor()]

        pipelines = {
            'video_frames': video_pipeline,
            'label': label_pipeline,
            'frame_index':frame_idx_pipeline,
            'video_idx':video_idx_pipeline
        }
        
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.TRAIN.FFCV.ENABLE:
            logger.info("Start FFCV dataloader for training.")
            write_path = os.path.join(cfg.TRAIN.FFCV.DATAPATH_PREFIX,cfg.TRAIN.SPLIT)+".beton"
            logger.info("Load the FFCV beton file for training:{}".format(write_path))
            if cfg.NUM_GPUS > 1:
                train_loader = Loader(write_path, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.RANDOM, pipelines=pipelines,distributed = True,os_cache=cfg.DATA_LOADER.OS_CACHE)
            else:
                train_loader = Loader(write_path, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.QUASI_RANDOM, pipelines=pipelines,distributed = False,os_cache=cfg.DATA_LOADER.OS_CACHE)
            num_videos  = train_loader.first_traversal_order.shape[0]
            logger.info("The FFCV beton file has {} samples for training.".format(num_videos))
            train_meter = TrainMeter(len(train_loader), cfg)
  
        if cfg.VAL.ENABLE and cfg.VAL.FFCV.ENABLE:  
            logger.info("Start FFCV dataloader for validation.")
            write_path = os.path.join(cfg.VAL.FFCV.DATAPATH_PREFIX,cfg.VAL.SPLIT)+".beton"
            logger.info("Load the FFCV beton file for validation:{}".format(write_path))
            if cfg.NUM_GPUS > 1:
                val_loader = Loader(write_path, batch_size=cfg.VAL.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.RANDOM, pipelines=pipelines,distributed = True,os_cache=cfg.DATA_LOADER.OS_CACHE)
            else:
                val_loader = Loader(write_path, batch_size=cfg.VAL.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.QUASI_RANDOM, pipelines=pipelines,distributed = False,os_cache=cfg.DATA_LOADER.OS_CACHE)
            num_videos  = val_loader.first_traversal_order.shape[0]
            logger.info("The FFCV beton file has {} samples for validation.".format(num_videos))
            val_meter = ValMeter(len(val_loader), cfg)  

             
        # Train for one epoch.
        epoch_timer.epoch_tic()

        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )

        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        
        # Save a checkpoint.
        if is_checkp_epoch and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            states = {
            'epoch': cur_epoch + 1,
            'model': cfg.MODEL.MODEL_NAME,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
            dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
            name = "checkpoint_epoch_{:05d}.pyth".format(cur_epoch + 1)
            save_ck_path = os.path.join(dir, name)
            torch.save(states, save_ck_path)

        # validation
        if cfg.VAL.ENABLE:
            is_eval_epoch = (
                misc.is_eval_epoch(
                    cfg,
                    cur_epoch,
                    None,
                )
            )
            # Evaluate the model on validation set.
            if is_eval_epoch:
                eval_epoch(
                    val_loader,
                    model,
                    val_meter,
                    cur_epoch,
                    cfg,
                    train_loader,
                    last_epoch = True if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH else False
                )
    
    logger.info("training done!")
