import os
from datasets.base_dataset_fsss import BASE_DATASET_FSSS
from datasets.dataset_split import VISION_V1_split_train_dict, VISION_V1_split_test_dict

from datasets.utilis_data import obtain_filename_segmentation_category_dict, generate_category_filename
from utils.multiprocessing import is_master_proc
import logging
import pickle

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VISION_V1_FSSS(BASE_DATASET_FSSS):

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
            target_split_list = VISION_V1_split_train_dict[self.data_split]
            LOGGER.info("current training class is {}".format(target_split_list))
        else:
            target_split_list = VISION_V1_split_test_dict[self.data_split]
            LOGGER.info("current test class is {}".format(target_split_list))

        # initialize dir
        init_dir = os.listdir(self.source)
        init_dir = [os.path.join(self.source, i)
                    for i in init_dir if i[0] != "." and os.path.isdir(os.path.join(self.source, i))]

        total_name = []
        target_name = ["object_filename", "object_category_filename", "object_category_filename_list"]
        mode_name = "train" if self.mode == "train" else "test"
        for each_target_name in target_name:
            _name = "_".join(["vision_dataset", mode_name, self.data_split, each_target_name, ".pkl"])
            total_name.append(os.path.join(cfg.DATASET.vision_data_save_path, _name))

        if cfg.DATASET.vision_data_load:
            class_names_ = self.__dict__
            for idx, each_target_name in enumerate(total_name):
                with open(each_target_name, 'rb') as f:
                    class_names_[target_name[idx]] = pickle.load(f)
        else:
            self.object_filename, self.object_category_filename, self.object_category_filename_list = \
                self.get_basic_element(init_dir)
            class_names_ = self.__dict__

            if cfg.DATASET.vision_data_save:
                if is_master_proc():
                    for idx, each_target_name in enumerate(total_name):
                        with open(each_target_name, 'wb') as f:
                            pickle.dump(class_names_[target_name[idx]], f)

    def get_basic_element(self, init_dir):
        object_category_filename = {}
        object_filename = {}
        object_category_filename_list = []

        if "train" in self.mode:
            target_split_list = VISION_V1_split_train_dict[self.data_split]
        else:
            target_split_list = VISION_V1_split_test_dict[self.data_split]

        for tmp_path in init_dir:
            tmp_object = tmp_path.split("/")[-1]

            if tmp_object in target_split_list:
                train_path = os.path.join(tmp_path, "train")
                val_path = os.path.join(tmp_path, "val")

                filename_segmentation_category, category_num = obtain_filename_segmentation_category_dict(train_path)
                filename_segmentation_category_val, category_num = obtain_filename_segmentation_category_dict(val_path)

                filename_segmentation_category.update(filename_segmentation_category_val)

                object_filename[tmp_object] = filename_segmentation_category

                category_filename = generate_category_filename(category_num, filename_segmentation_category)

                object_category_filename[tmp_object] = category_filename

                for category, filename in category_filename.items():
                    for tmp_filename in filename:
                        object_category_filename_list.append("^".join([tmp_object, str(category), tmp_filename]))

        return object_filename, object_category_filename, object_category_filename_list
