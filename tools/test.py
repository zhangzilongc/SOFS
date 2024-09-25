import pprint
import numpy as np
import torch
import logging
import traceback

from utils import init_distributed_training, is_master_proc, seed_everything, \
    freeze_paras, get_datasets, setup_logging

from model.SegGPT.model_seggpt import prepare_model

from model.SOFS import SOFS

from tools.epoch_train_eval_ss import epoch_validate_ss, epoch_validate_non_resize_ss
from tools.open_domain_eval import opendomain_eval
from tools.contest_eval import eccv_contest_eval

import matplotlib

matplotlib.use('Agg')

from torch.utils.data.sampler import SubsetRandomSampler
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER = logging.getLogger(__name__)


def test(cfg):
    """
    support test in multiple GPUs and a single GPU
    """
    # distributed test
    # set up environment
    init_distributed_training(cfg)

    # Assign different random seeds. Sharing random seeds may cause data generated by each process to be consistent.
    if torch.distributed.is_initialized():
        temp_rank = torch.distributed.get_rank()
    else:
        temp_rank = 0

    # setup logging
    if is_master_proc():
        setup_logging(cfg)
        LOGGER.info(pprint.pformat(cfg))

    if cfg.NUM_GPUS == 1:
        torch.cuda.set_device(cfg.DEVICE)

    try:
        LOGGER.info("start test!")
        cur_device = torch.cuda.current_device()

        # multiple runs for few-shot learning to reduce the random
        rand_seed_list = [i + cfg.RNG_SEED for i in range(cfg.DATASET.few_shot_repeated_multiple)]

        if cfg.DATASET.name in ["visa", "mvtec"]:
            final_result_collect = {
                "AUROC": [],
                "mean_AP_sample": [],
                "f1_score_max_sample": [],
                "Pixel_AUROC": [],
                "Mean_AP_Pixel": [],
                "f1_score_max": []
            }

            mean_result_collect = {i: {
                "AUROC": [],
                "mean_AP_sample": [],
                "f1_score_max_sample": [],
                "Pixel_AUROC": [],
                "Mean_AP_Pixel": [],
                "f1_score_max": []
            } for i in cfg.DATASET.sub_datasets}

            result_collect = {i: {"AUROC": [],
                                  "mean_AP_sample": [],
                                  "f1_score_max_sample": [],
                                  "Pixel_AUROC": [],
                                  "Mean_AP_Pixel": [],
                                  "f1_score_max": []} for i in rand_seed_list}
        elif cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND",
                                  "DS_Spectrum_DS", "DS_Spectrum_DS_ND",
                                  "opendomain_test_dataset_ND", "ECCV_Contest_Test_ND"]:
            pass
        else:
            raise NotImplementedError

        for tmp_rand_seed in rand_seed_list:
            # Set random seed from configs.
            seed_everything(tmp_rand_seed + temp_rank)

            LOGGER.info("load dataset!")
            test_datasets = get_datasets(cfg=cfg, mode='test')
            LOGGER.info("load complete!")

            # training process
            for idx, individual_datasets in enumerate(test_datasets):
                LOGGER.info("current dataset is {}.".format(individual_datasets.name))
                LOGGER.info(
                    "the data in current dataset {} are {}.".format(individual_datasets.name, len(individual_datasets)))

                # start training
                torch.cuda.empty_cache()

                if cfg.NUM_GPUS > 1:
                    test_sampler = torch.utils.data.distributed.DistributedSampler(individual_datasets)
                    test_loader = torch.utils.data.DataLoader(
                        individual_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                        num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, sampler=test_sampler,
                        drop_last=True)
                else:
                    test_loader = torch.utils.data.DataLoader(
                        individual_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                        num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True)

                LOGGER.info("load model!")
                if cfg.TEST.method == "SegGPT":
                    model = prepare_model()
                elif cfg.TEST.method == "SOFS":
                    model = SOFS(cfg=cfg)
                else:
                    raise NotImplementedError("test method is not in the target list!")

                if cfg.TEST.load_checkpoint:
                    save_checkpoint = torch.load(cfg.TEST.load_model_path,
                                                 map_location="cpu")
                    model.load_state_dict(save_checkpoint, strict=False)
                    LOGGER.info("load main model successful!")

                freeze_paras(model)

                model = model.cuda(cur_device)

                # start test
                torch.cuda.empty_cache()

                """
                                    prediction
                                ______1________0____
                              1 |    TP   |   FN   |
                ground truth  0 |    FP   |   TN   |
    
                ACC = (TP + TN) / (TP + FP + FN + TN)
    
                precision = TP / (TP + FP)
    
                recall (TPR) = TP / (TP + FN)
    
                FPR（False Positive Rate）= FP / (FP + TN)
                """
                if cfg.NUM_GPUS > 1:
                    test_sampler.set_epoch(tmp_rand_seed)

                if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND",
                                        "DS_Spectrum_DS", "DS_Spectrum_DS_ND",
                                        "opendomain_test_dataset_ND", "ECCV_Contest_Test_ND"]:
                    if cfg.TEST.method in ["SegGPT", "SOFS"]:
                        if cfg.DATASET.name in ["VISION_V1_ND", "DS_Spectrum_DS_ND"]:
                            epoch_validate_non_resize_ss(
                                val_loader=test_loader,
                                model=model,
                                epoch=1,
                                cfg=cfg,
                                rand_seed=tmp_rand_seed
                            )
                        elif cfg.DATASET.name in ["VISION_V1", "DS_Spectrum_DS"]:
                            epoch_validate_ss(
                                val_loader=test_loader,
                                model=model,
                                epoch=1,
                                cfg=cfg,
                                rand_seed=tmp_rand_seed
                            )
                        elif cfg.DATASET.name in ["opendomain_test_dataset_ND"]:
                            opendomain_eval(
                                val_loader=test_loader,
                                model=model,
                                epoch=1,
                                cfg=cfg,
                                rand_seed=tmp_rand_seed
                            )
                        elif cfg.DATASET.name in ["ECCV_Contest_Test_ND"]:
                            eccv_contest_eval(
                                val_loader=test_loader,
                                model=model,
                                epoch=1,
                                cfg=cfg,
                                rand_seed=tmp_rand_seed
                            )
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                elif cfg.DATASET.name in ["visa", "mvtec"]:
                    if cfg.TEST.method in ["SOFS"]:
                        # todo
                        pass
                    else:
                        raise NotImplementedError

        LOGGER.info("Method test phase complete!")
    except Exception as e:
        LOGGER.error("error：")
        LOGGER.error(e)
        LOGGER.error("\n" + traceback.format_exc())
