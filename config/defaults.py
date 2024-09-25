#!/usr/bin/env python3

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# multiple GPU
_C.NUM_GPUS = 8
_C.RNG_SEED = 54
# for a single GPU
_C.DEVICE = 3
_C.DIST_BACKEND = "nccl"  # Distributed backend.

_C.OUTPUT_DIR = './log_few_shot_IVI'

_C.DATASET = CfgNode()
_C.DATASET.name = 'vision_dataset'
_C.DATASET.split = 0
_C.DATASET.image_size = 518
_C.DATASET.mask_size = 518
_C.DATASET.unified_mask_size = 4000
_C.DATASET.rotate_min = -10
_C.DATASET.rotate_max = 10
# for non_resize crop dataset
_C.DATASET.area_resize_ratio = 0.01
_C.DATASET.crop_size = 518
_C.DATASET.crop_ratio = [0.7, 1.2]
# number of repeated samples in every test sample
_C.DATASET.test_sample_repeated_multiple = 1
_C.DATASET.few_shot_repeated_multiple = 1
_C.DATASET.shot = 1
# only for semantic segmentation, sample multiple crop in one image
_C.DATASET.s_in_shot = 1
# for anomaly detection
_C.DATASET.sub_datasets = ["original"]
_C.DATASET.transform_length_width_ratio = True
_C.DATASET.vision_data_save = True
_C.DATASET.vision_data_save_path = "/usr/sdc/zzl/vision_data"
_C.DATASET.vision_data_load = False
# alpha in the SOFS paper
_C.DATASET.normal_sample_sampling_prob = 0.3

_C.DATASET.open_domain_test_object = ["severstal_steel"]
_C.DATASET.open_domain_object_category_num = [1]
_C.DATASET.open_domain_specific_defect_category = [0]  # 0 is the first

_C.TRAIN = CfgNode()
_C.TRAIN.enable = True
_C.TRAIN.save_model = False
_C.TRAIN.method = 'SOFS'
_C.TRAIN.backbone = 'dinov2_vitb14'
_C.TRAIN.backbone_load_state_dict = True
_C.TRAIN.backbone_checkpoint = './dinov2_vitb14_pretrain.pth'
_C.TRAIN.dataset_path = '/usr/sdd/zzl_data/defect_detection/vision_dataset'
_C.TRAIN.load_checkpoint = False
_C.TRAIN.load_model_path = "./save_model"

_C.TRAIN.SOFS = CfgNode()
# layer to use in DINO V2
_C.TRAIN.SOFS.prior_layer_pointer = [5, 6, 7, 8, 9, 10]
_C.TRAIN.SOFS.target_semantic_temperature = 0.1

# self attention setups
_C.TRAIN.SOFS.transformer_nums_heads = 4
_C.TRAIN.SOFS.transformer_num_stages = 2
# correspond to C in the SOFS paper
_C.TRAIN.SOFS.reduce_dim = 256
# correspond to C1 in the SOFS paper
_C.TRAIN.SOFS.transformer_embed_dim = 256

# for the designs for SOFS
_C.TRAIN.SOFS.meta_cls = True
# for abnormal prior map
_C.TRAIN.SOFS.normal_sim_aug = True
# prototype intensity downsampling
_C.TRAIN.SOFS.conv_vit_down_sampling = True
# for l in the prototype intensity downsampling
_C.TRAIN.SOFS.vit_patch_size = 14
# eta in the paper
_C.TRAIN.SOFS.smooth_r = 1e5

# todo in the next version
_C.TRAIN.LOSS = CfgNode()
_C.TRAIN.LOSS.dice_weight = 1.
_C.TRAIN.LOSS.ce_weight = 0.01

_C.TRAIN_SETUPS = CfgNode()
_C.TRAIN_SETUPS.batch_size = 4
_C.TRAIN_SETUPS.num_workers = 8
_C.TRAIN_SETUPS.learning_rate = 1e-5
_C.TRAIN_SETUPS.epochs = 50
_C.TRAIN_SETUPS.optimizer_momentum = 0.9
_C.TRAIN_SETUPS.weight_decay = 0.01
_C.TRAIN_SETUPS.poly_training = True
_C.TRAIN_SETUPS.lr_multiple = 2.

_C.TRAIN_SETUPS.TEST_SETUPS = CfgNode()
_C.TRAIN_SETUPS.TEST_SETUPS.test_state = True
_C.TRAIN_SETUPS.TEST_SETUPS.epoch_test = 50
_C.TRAIN_SETUPS.TEST_SETUPS.train_miou = 50
_C.TRAIN_SETUPS.TEST_SETUPS.val_state = False

_C.TEST = CfgNode()
_C.TEST.enable = False
_C.TEST.method = 'SOFS'
_C.TEST.dataset_path = '/usr/sdd/zzl_data/defect_detection/vision_dataset'
_C.TEST.load_checkpoint = True
_C.TEST.load_model_path = "./save_model/xxxx.pth"

# threshold for SegGPT
_C.TEST.semantic_threshold = 0.6

_C.TEST.VISUALIZE = CfgNode()
_C.TEST.VISUALIZE.save_figure = True
# the prob for saving figure
_C.TEST.VISUALIZE.sample_prob = 0.1

_C.TEST_SETUPS = CfgNode()
_C.TEST_SETUPS.batch_size = 4

_C.TEST_SETUPS.ND_batch_size = 4


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
