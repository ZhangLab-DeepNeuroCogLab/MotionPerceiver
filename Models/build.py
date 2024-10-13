#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import torch
import Mainframe.set_logging as set_logging
from Models.MP import MP
from Models.En_MP import En_MP

logger = set_logging.get_logger(__name__)

DATASET_CLASSES = {
    'ntu_rgbd60': 60,
    "bmp":10,
    "nw_ucla": 10,
    "none":0
}

def add_prefix_state_dict(checkpoint_path, prefix='module.'):
    checkpoint = torch.load(checkpoint_path)
    original_state_dict = checkpoint['state_dict']
    modified_state_dict = {prefix + key: value for key, value in original_state_dict.items()}
    checkpoint['state_dict'] = modified_state_dict
    return checkpoint

def remove_prefix_state_dict(checkpoint_path, prefix='module.'):
    checkpoint = torch.load(checkpoint_path,map_location="cpu")
    original_state_dict = checkpoint['state_dict']
    modified_state_dict = {key[len(prefix):]: value for key, value in original_state_dict.items() if key.startswith(prefix)}
    checkpoint['state_dict'] = modified_state_dict
    return checkpoint
    
    
def build_model(cfg, mode, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    
    if cfg.TRAIN.PRETRAIN_DATASET is not None:
        n_finetune_classes = DATASET_CLASSES[cfg.TRAIN.DATASET]
        n_pretrain_classes = DATASET_CLASSES[cfg.TRAIN.PRETRAIN_DATASET]
    else:
        n_finetune_classes = DATASET_CLASSES[cfg.TRAIN.DATASET]
        n_pretrain_classes = None
    
    if cfg.MODEL.MODEL_NAME == "MP":
        model = MP(cfg,mode,backbone = cfg.MYMODEL.BACKBONE,n_pretrain_classes=n_pretrain_classes,n_finetune_classes=n_finetune_classes)      
    elif cfg.MODEL.MODEL_NAME == "En_MP":
        model = En_MP(cfg,mode,backbone = cfg.MYMODEL.BACKBONE,n_pretrain_classes=n_pretrain_classes,n_finetune_classes=n_finetune_classes)      
    
    if cfg.NUM_GPUS:
        if gpu_id is None:
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        model = model.cuda(device=cur_device)
        
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=False,
        )
        
    if mode =="test":
        if cfg.NUM_GPUS == 1 and cfg.DDP_CKPT: 
            trained_model = remove_prefix_state_dict(cfg.TEST.CHECKPOINT_FILE_PATH)    
        elif cfg.NUM_GPUS > 1 and cfg.DDP_CKPT:
            trained_model = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH,map_location="cpu")
        elif cfg.NUM_GPUS == 1 and not cfg.DDP_CKPT: 
            trained_model = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH)
        elif cfg.NUM_GPUS > 1 and not cfg.DDP_CKPT: 
            trained_model = add_prefix_state_dict(cfg.TEST.CHECKPOINT_FILE_PATH)    
        logger.info("TEST CHECKPOINT FILE PATH is {}".format(cfg.TEST.CHECKPOINT_FILE_PATH))
        model.load_state_dict(trained_model['state_dict'])   
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH and "torchhub" not in cfg.TRAIN.CHECKPOINT_FILE_PATH:
        if cfg.NUM_GPUS == 1 and cfg.DDP_CKPT:   
            trained_model = remove_prefix_state_dict(cfg.TRAIN.CHECKPOINT_FILE_PATH)  
        elif cfg.NUM_GPUS > 1 and cfg.DDP_CKPT:
            trained_model = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH,map_location="cpu")
        elif cfg.NUM_GPUS == 1 and not cfg.DDP_CKPT: 
            trained_model = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH)
        elif cfg.NUM_GPUS > 1 and not cfg.DDP_CKPT: 
            trained_model = add_prefix_state_dict(cfg.TRAIN.CHECKPOINT_FILE_PATH)    
        model.load_state_dict(trained_model['state_dict'])     
        logger.info("TRAIN CHECKPOINT FILE PATH is {}".format(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        
    return model
