#!/usr/bin/python3

import logging
import os

from utils import load_config, parse_args, launch_job

from tools import train, test

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:
        # merge config and args, mkdir image_save and checkpoints
        cfg = load_config(args, path_to_config=path_to_config)

        # Perform training and test in each category.
        if cfg.TRAIN.enable:
            """
            todo in the new version
            include:
             1) train and test dataloader load
             2) training prepare phase: 1) base model load
                                        2) optimizer load
             3) start training: include various methods (class module)
             4) complete training: start test (one follow by one)
            """
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        if cfg.TEST.enable:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
