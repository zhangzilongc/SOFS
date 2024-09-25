"""Argument parser functions."""

import argparse
import sys
import os

from config import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser.
    """
    parser = argparse.ArgumentParser(
        description="Provide Training and Test pipeline for Vision-based Industrial Inspection."
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="the device to train/test model",
        default="0",
    )

    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["./method_config/DS_Spectrum/SOFS.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--prior_layer_pointer",
        help="the layer used in the backbone",
        default=[5, 6, 7, 8, 9, 10],
        nargs="+",
    )
    # add from command line
    parser.add_argument(
        "--opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id

    prior_layer_pointer = args.prior_layer_pointer
    prior_layer_pointer = [int(i) for i in prior_layer_pointer]

    if cfg.TRAIN.method in ["SOFS"] or cfg.TEST.method in ["SOFS"]:
        method = "SOFS"
        exec("cfg.TRAIN.{}.prior_layer_pointer=prior_layer_pointer".format(method))

    method = cfg.TRAIN.method

    # Create the checkpoint dir. ./checkpoints
    cfg.OUTPUT_DIR = "_".join([
        cfg.OUTPUT_DIR,
        cfg.DATASET.name,
        method,
        str(cfg.RNG_SEED)
    ])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return cfg
