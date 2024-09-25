import logging


LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "VISION_V1": ["datasets.vision_v1_fsss", "VISION_V1_FSSS"],
    "VISION_V1_ND": ["datasets.vision_v1_fsss_ND", "VISION_V1_FSSS_ND"],
    "DS_Spectrum_DS": ["datasets.ds_spectrum_DS_fsss", "DS_Spectrum_DS_FSSS"],
    "DS_Spectrum_DS_ND": ["datasets.ds_spectrum_DS_fsss_ND", "DS_Spectrum_DS_FSSS_ND"],
    "mvtec": ["datasets.few_shot_ad", "Few_Shot_AD_Dataset"],
    "visa": ["datasets.few_shot_ad", "Few_Shot_AD_Dataset"],
    "opendomain_test_dataset_ND": ["datasets.opendomain_test_ND", "OpenDomain_Test_ND"],
    "ECCV_Contest_ND": ["datasets.contest_fsss_ND", "CONTEST_FSSS_ND"],
    "ECCV_Contest_Test_ND": ["datasets.contest_test_ND", "CONTEST_Test_ND"]
}


def get_datasets(cfg, mode='train'):
    datasets_list = []
    dataset_info = _DATASETS[cfg.DATASET.name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    if mode == "train":
        dataset = dataset_library.__dict__[dataset_info[1]](
            cfg=cfg,
            mode=mode
        )

        dataset.name = cfg.DATASET.name
        dataset.name += "_" + "original"

        datasets_list.append(dataset)
    else:
        for sub_dataset in cfg.DATASET.sub_datasets:
            dataset = dataset_library.__dict__[dataset_info[1]](
                cfg=cfg,
                mode=mode,
                classname=sub_dataset
            )

            dataset.name = cfg.DATASET.name
            dataset.name += "_" + sub_dataset

            datasets_list.append(dataset)

    return datasets_list
