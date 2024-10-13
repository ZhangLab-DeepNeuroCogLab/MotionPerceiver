#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson

import Utils.distributed as du
from Utils.env import pathmgr


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # Use 1K buffer if writing to cloud storage.
    io = pathmgr.open(
        filename, "a", buffering=1024 if "://" in filename else -1
    )
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None,overwrite=False):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    Args:
        output_dir: output directory of logging and json files.
        overwrite: whether to overwrite the existed logging and json files.
    """

    if du.is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    # setting the root logger such that children logger can inherit its property
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) 
    logger.propagate = False
    
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    # Output LogRecord into the stdout stream
    if du.is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    # Output LogRecord into a file
    if output_dir is not None and du.is_master_proc(du.get_world_size()):
        filename = os.path.join(output_dir, "stdout.log") 

        if overwrite:
            fh = logging.FileHandler(filename,mode="w") # overwrite test of the existed log
        else:
            fh = logging.FileHandler(filename) # append text of the existed log

        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats, output_dir=None):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
    if du.is_master_proc(du.get_world_size()) and output_dir:
        filename = os.path.join(output_dir, "json_stats.log")
        try:
            with pathmgr.open(
                filename, "a", buffering=1024 if "://" in filename else -1
            ) as f:
                f.write("json_stats: {:s}\n".format(json_stats))
        except Exception:
            logger.info(
                "Failed to write to json_stats.log: {}".format(json_stats)
            )
