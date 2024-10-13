#!/usr/bin/env python3

import sys
import os


"""Wrapper to train and test a model."""
import warnings
warnings.filterwarnings("ignore") # Ignore all warnings

from Config.defaults import assert_and_infer_cfg
from Utils.misc import launch_job
from Utils.parser import load_config, parse_args

from test_net import test
from train_net import train
import Mainframe.set_logging as set_logging
import torch
import random
import numpy as np
import sys
import os
import shutil

logger = set_logging.get_logger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

def main():
    """
    Main function to spawn the train and test process.
    """
    
    args = parse_args() 
    for path_to_config in args.cfg_files: 
        cfg = load_config(args, path_to_config) 
        cfg = assert_and_infer_cfg(cfg)

        # Setup (root) logging format (and inherit it).
        set_logging.setup_logging(cfg.OUTPUT_DIR,overwrite=True)  
        logger.info("config files: {}".format(args.cfg_files))

        # Set random seed from configs.
        set_seed(cfg.RNG_SEED)

        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=cfg.INIT_METHOD, func=train)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            assert cfg.TEST.NUM_ENSEMBLE_VIEWS >= 1
            assert cfg.TEST.NUM_SPATIAL_CROPS >= 1
            launch_job(cfg=cfg, init_method=cfg.INIT_METHOD, func=test)



if __name__ == "__main__":
    main()
