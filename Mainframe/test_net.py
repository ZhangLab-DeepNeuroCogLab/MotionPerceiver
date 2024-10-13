#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import Utils.checkpoint as cu
import Utils.distributed as du
import Mainframe.set_logging as set_logging
import Utils.misc as misc
from Models import build_model
from Utils.env import pathmgr
from Utils.meters import TestMeter
import cv2
import ipdb
import decord 
from decord import cpu, gpu

from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField,IntField,TorchTensorField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder,NDArrayDecoder, RandomResizedCropRGBImageDecoder
from ffcv.pipeline.operation import Operation
from typing import List

from itertools import combinations
import random
import matplotlib.pyplot as plt
import torchshow as ts
import math

logger = set_logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, frame_index,video_idx) in enumerate(
        test_loader
    ):  
        inputs = inputs.float()/255.0 
        inputs = (inputs - torch.tensor(cfg.DATA.MEAN))/(torch.tensor(cfg.DATA.STD))
        
        if inputs.size(1) != cfg.DATA.NUM_FRAMES:
            logger.info(
            "Video is padded from {} frames to {} frames.".format(inputs.size(1),cfg.DATA.NUM_FRAMES)
            )
            B,T,H,W,C = inputs.size()
            indices = np.linspace(0, T, cfg.DATA.NUM_FRAMES,endpoint=False).astype(int)
            inputs = inputs[:,indices]
      
        inputs = inputs.permute(0, 4, 1, 2, 3)
                   
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.squeeze(1).cuda()
            video_idx = video_idx.cuda()
        test_meter.data_toc()

        _,preds = model(inputs,labels)
        
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )     
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
        
    # Log epoch stats and print the final testing results.
    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()
    if writer is not None:
        writer.plot_eval(preds=all_preds, labels=all_labels)

    save_result_name = "result_"+ cfg.TEST.SPLIT + ".pkl"
    save_path = os.path.join(cfg.OUTPUT_DIR, save_result_name)

    if du.is_root_proc():
        with pathmgr.open(save_path, "wb") as f:
            pickle.dump([all_preds, all_labels], f)

    logger.info(
        "Successfully saved prediction results to {}".format(save_path)
    )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)

    # Setup logging format.
    set_logging.setup_logging(cfg.OUTPUT_DIR)

    test_meters = []
    
    if cfg.TEST.CHECKPOINT_FILE_PATH == "":
        dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        names = pathmgr.ls(dir) if pathmgr.exists(dir) else []
        names = [f for f in names if f.startswith("checkpoint")]
        assert len(names) > 0, "NO CHECKPOINTS CAN BE LOADED !!!"
        name = sorted(names)[-1]
        cfg.TEST.CHECKPOINT_FILE_PATH = os.path.join(dir, name)
        logger.info("TEST CHECKPOINT FILE PATH is {}".format(cfg.TEST.CHECKPOINT_FILE_PATH))
        
    model = build_model(cfg,"test")

    if cfg.TEST.FFCV.ENABLE:
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
        
        write_path = os.path.join(cfg.TEST.FFCV.DATAPATH_PREFIX,cfg.TEST.SPLIT)+".beton"
        if cfg.NUM_GPUS > 1:
            test_loader = Loader(write_path, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.SEQUENTIAL, distributed = True,pipelines=pipelines,drop_last=False)
            test_loader_for_num_videos = Loader(write_path, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.SEQUENTIAL, distributed = False,pipelines=pipelines,drop_last=False)
            num_videos  = test_loader_for_num_videos.first_traversal_order.shape[0]
        else:
            test_loader = Loader(write_path, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, order=OrderOption.SEQUENTIAL, distributed = False,pipelines=pipelines,drop_last=False)
            num_videos  = test_loader.first_traversal_order.shape[0]
        logger.info("Testing dataset beton path is {}".format(write_path))
    
    logger.info("Testing model for {} iterations with {} samples".format(len(test_loader),num_videos))

    assert (
        num_videos
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )

    test_meter = TestMeter(
        num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):  
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    logger.info(
        "Finalized testing with {} temporal clips and {} spatial crops".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        )
    )

    result_string = (
        "Test Top1 Acc: {}% Test Top5 Acc: {}%"
        "".format(
            test_meter.stats["top1_acc"],
            test_meter.stats["top5_acc"],
        )
    )
    logger.info("{}".format(result_string))
    
    return result_string 
