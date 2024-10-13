#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
import math
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True, Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Kill training if loss explodes over this ratio from the previous 5 measurements.
# Only enforced if > 0.0.
_C.TRAIN.KILL_LOSS_EXPLOSION_FACTOR = 0.0

# Dataset name for training.
_C.TRAIN.DATASET = ""

# Dataset split for training.
_C.TRAIN.SPLIT = ""

# Pre-train Dataset.
_C.TRAIN.PRETRAIN_DATASET = ""

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, use FP16 for activations.
_C.TRAIN.MIXED_PRECISION = True

# -----------------------------------------------------------------------------
# FFCV options for training
# -----------------------------------------------------------------------------
_C.TRAIN.FFCV = CfgNode()

# If True, use FFCV for data loading during training.
_C.TRAIN.FFCV.ENABLE  = False

# Datapath prefix of FFCV beton files
_C.TRAIN.FFCV.DATAPATH_PREFIX = ""

# ---------------------------------------------------------------------------- #
# Validation options.
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

# If True Validate the model, else skip validation.
_C.VAL.ENABLE = True

# Dataset for validation.
_C.VAL.DATASET = ""

# Evaluate model every eval period epochs.
_C.VAL.EVAL_PERIOD = 10

# Total mini-batch size.
_C.VAL.BATCH_SIZE = 64

# Dataset split for validation.
_C.VAL.SPLIT = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.VAL.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.VAL.NUM_SPATIAL_CROPS = 3

# -----------------------------------------------------------------------------
# FFCV options for validation
# -----------------------------------------------------------------------------
_C.VAL.FFCV = CfgNode()

# If True, use FFCV for data loading during validation.
_C.VAL.FFCV.ENABLE  = False

# Datapath prefix of FFCV beton files
_C.VAL.FFCV.DATAPATH_PREFIX = ""

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset name for testing.
_C.TEST.DATASET = ""

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results for testing.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Dataset split for testing.
_C.TEST.SPLIT = ""


# -----------------------------------------------------------------------------
# FFCV options for testing
# -----------------------------------------------------------------------------
_C.TEST.FFCV = CfgNode()

# If True, use FFCV for data loading during validation.
_C.TEST.FFCV.ENABLE  = False

# Datapath prefix of FFCV beton files
_C.TEST.FFCV.DATAPATH_PREFIX  = ""

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = ""

# Model modality
_C.MODEL.MODALITY = 1

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# -----------------------------------------------------------------------------
# Motion Perceiver options
# -----------------------------------------------------------------------------
_C.MYMODEL = CfgNode()

# The feture extraction backbone used in the Motion Perceiver 
_C.MYMODEL.BACKBONE = "dino_vitb16" 

# Temperature in the slot contrastive loss 
_C.MYMODEL.TEMPORAL_CORR_TEMP = 0.05

# Number of frames used to estimate optical flow for each token (multiscale as a list)
_C.MYMODEL.NUM_FRAMES_LIST = []

# -----------------------------------------------------------------------------
# Slot Attention Module options
# -----------------------------------------------------------------------------
_C.MYMODEL.TEMPORAL_SLOT = CfgNode()

# Number of iterations
_C.MYMODEL.TEMPORAL_SLOT.NUM_ITER = 3

# Number of slots
_C.MYMODEL.TEMPORAL_SLOT.NUM_SLOTS = []

# Dimension size of input channel
_C.MYMODEL.TEMPORAL_SLOT.INPUT_CHANNEL = []

# Dimension size of slots
_C.MYMODEL.TEMPORAL_SLOT.SLOT_SIZE = []

# Dimension size of MLP hidden layer
_C.MYMODEL.TEMPORAL_SLOT.MLP_HIDDEN_SIZE = []

# Number of head in slots
_C.MYMODEL.TEMPORAL_SLOT.NUM_HEAD = 1

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# The spatial crop size
_C.DATA.CROP_SIZE = 224

# The resize scale of frames in training (better to keep the aspect ratio).
_C.DATA.RESIZE_SCALE_WIDTH = 224
_C.DATA.RESIZE_SCALE_HEIGHT =300

# The mean value of the video raw pixels across the RGB channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the RGB channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Number of frames in each video
_C.DATA.NUM_FRAMES = 32

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# LARS optimizer
_C.SOLVER.LARS_ON = False

# Adam's beta
_C.SOLVER.BETAS = (0.9, 0.999)

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Initialization method, includes TCP or shared file-system
_C.INIT_METHOD = "tcp://localhost:9999"

# whether the checkpoint used is from distributed data parallel mechanism
_C.DDP_CKPT = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training/testing process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to CPU memory.
_C.DATA_LOADER.OS_CACHE = False

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True



def assert_and_infer_cfg(cfg):
    
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
