import os
from datasets.base_dataset_fsss import BASE_DATASET_FSSS
from datasets.dataset_split import DS_Spectrum_split_train_dict, DS_Spectrum_split_test_dict

from datasets.utilis_data import obtain_filename_segmentation_category_dict, generate_category_filename
from utils.multiprocessing import is_master_proc
import logging
import numpy as np
from glob import glob
import PIL.Image as PIL_Image

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DS_Spectrum_FSSS(BASE_DATASET_FSSS):

    def __init__(
            self,
            cfg,
            mode="train",
            **kwargs
    ):
        super().__init__(cfg=cfg, mode=mode, kwargs=kwargs)

        """
        object_filename, object_category_filename and object_category_filename_list are shared for all dataset

        object_filename:
            segmented object: the object for semantic segmentation
            path1 of original image: self.source + path1 of original image = the absolute path to the file
            seg: seg includes every binary (bool) segmentation for target semantic, the shape of every segmentation is (original_image_h, original_image_w)
            category: corresponding class for each ele in seg
            category_sum: the total category for path1 of original image

            object_filename = {
                "segmented object": {
                    "path1 of original image": {
                        "seg": [np.array([[False, False, ..., False], [...]]) (bool), np.array([[False, False, ..., False], [...]]) (bool)],
                        "category": [0, 1],
                        "category_sum": 2
                    },
                    "path2 of original image": {
                        "seg": [np.array((h, w))(bool)],
                        "category": [2],
                        "category_sum": 1
                    }
                }
            }

            e.g. source + image/glue/003.png = /usr/xxx/image/glue/003.png
            object_filename = {
                "grid": {
                    "image/glue/003.png": {
                        "seg": [np.array([[False, False, ..., False], [...], ...])],
                        "category": [2],
                        "category_sum": 1
                    }
                    }
                }
            }

        object_category_filename:
            every path for different category

            object_category_filename = {
                "segmented object": {
                    category(int): [
                        path1 of original image, 
                        path2 of original image,
                        ...
                    ]
                }
            }

            e.g.
            object_category_filename = {
                "grid": {
                    0(the first category): ["image/glue/003.png", ...]
                }
            }

        object_category_filename_list:
            "^" to join segmented object, category, path2 of original image

        e.g.
        object_category_filename_list = ["grid^0^image/glue/003.png", ...]
        """

        if self.mode == "train":
            target_split_list = DS_Spectrum_split_train_dict[self.data_split]
            LOGGER.info("current training class is {}".format(target_split_list))
        else:
            target_split_list = DS_Spectrum_split_test_dict[self.data_split]
            LOGGER.info("current test class is {}".format(target_split_list))

        # initialize dir
        init_dir = os.listdir(self.source)
        init_dir = [os.path.join(self.source, i)
                    for i in init_dir if i[0] != "." and os.path.isdir(os.path.join(self.source, i))]

        self.object_filename, self.object_category_filename, self.object_category_filename_list = \
            self.get_basic_element(init_dir)

    def get_basic_element(self, init_dir):
        object_category_filename = {}
        object_filename = {}
        object_category_filename_list = []

        if "train" in self.mode:
            target_split_list = DS_Spectrum_split_train_dict[self.data_split]
        else:
            target_split_list = DS_Spectrum_split_test_dict[self.data_split]

        for tmp_path in init_dir:
            tmp_object = tmp_path.split("/")[-1]

            if tmp_object in target_split_list:
                current_path = os.path.join(tmp_path, "image")
                if tmp_object in ["DS-Cotton-Fabric", "DS-DAGM",
                                  "Capacitor_VISION", "Console_VISION",
                                  "Groove_VISION", "Ring_VISION",
                                  "Screw_VISION", "Wood_VISION"]:
                    img_path = glob(os.path.join(current_path, "*.jpg"))
                    img_path += glob(os.path.join(current_path, "*.png"))
                else:
                    img_path = glob(os.path.join(current_path, "*/*.jpg"))
                    img_path += glob(os.path.join(current_path, "*/*.png"))

                filename_segmentation_category = {}
                for each_ele in img_path:
                    if each_ele[0] != "." and "good" not in each_ele:
                        if tmp_object in ["DS-Cotton-Fabric", "DS-DAGM",
                                          "Capacitor_VISION", "Console_VISION",
                                          "Groove_VISION", "Ring_VISION",
                                          "Screw_VISION", "Wood_VISION"
                                          ]:
                            ima_name_dict = "/".join(each_ele.split("/")[-2:])
                        else:
                            ima_name_dict = "/".join(each_ele.split("/")[-3:])

                        ima_name = each_ele.split("/")[-1].split(".")[0]
                        mask_path = ima_name_dict.replace("image", "mask")
                        if tmp_object in ["DS-Cotton-Fabric", "DS-DAGM",
                                          "Capacitor_VISION", "Console_VISION",
                                          "Groove_VISION", "Ring_VISION",
                                          "Screw_VISION", "Wood_VISION"]:
                            mask_path = "/".join(mask_path.split("/")[:1])
                        else:
                            mask_path = "/".join(mask_path.split("/")[:2])

                        mask_path = os.path.join(tmp_path, mask_path, ima_name + "_mask.png")

                        mask = np.array(PIL_Image.open(mask_path))
                        label_now = list(np.unique(mask)[1:])

                        for idx, tmp_label in enumerate(label_now):
                            tmp_mask = mask == tmp_label

                            if idx == 0:
                                filename_segmentation_category[ima_name_dict] = {
                                    "seg": [tmp_mask],
                                    "category": [int(tmp_label) - 1],
                                    'category_sum': len(label_now)
                                }
                            else:
                                filename_segmentation_category[ima_name_dict]["seg"].append(tmp_mask)
                                filename_segmentation_category[ima_name_dict]["category"].append(int(tmp_label) - 1)

                object_filename[tmp_object] = filename_segmentation_category

                # 类别：文件名 0: xxx
                if tmp_object == "DS-DAGM":
                    category_num = 5
                elif tmp_object == "DS-Cotton-Fabric":
                    category_num = 2
                elif tmp_object == "toothbrush":
                    category_num = 3
                elif tmp_object == "Capacitor_VISION":
                    category_num = 3
                elif tmp_object == "Console_VISION":
                    category_num = 4
                elif tmp_object == "Groove_VISION":
                    category_num = 2
                elif tmp_object == "Ring_VISION":
                    category_num = 4
                elif tmp_object == "Screw_VISION":
                    category_num = 3
                elif tmp_object == "Wood_VISION":
                    category_num = 4
                else:
                    category_num = len(
                        [i for i in os.listdir(current_path) if i[0] != "." and os.path.isdir(current_path)]) - 1

                category_filename = generate_category_filename(category_num, filename_segmentation_category)

                object_category_filename[tmp_object] = category_filename

                for category, filename in category_filename.items():
                    for tmp_filename in filename:
                        object_category_filename_list.append("^".join([tmp_object, str(category), tmp_filename]))

        return object_filename, object_category_filename, object_category_filename_list
