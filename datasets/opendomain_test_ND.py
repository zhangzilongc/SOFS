import os
from datasets.base_dataset_fsss_ND import BASE_DATASET_FSSS_ND
from datasets.dataset_split import VISION_V1_split_train_dict, VISION_V1_split_test_dict

from datasets.utilis_data import obtain_filename_segmentation_category_dict, generate_category_filename
from utils.multiprocessing import is_master_proc
import logging
import PIL.Image as PIL_Image
import numpy as np
import torch
import cv2
import random
from glob import glob

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OpenDomain_Test_ND(BASE_DATASET_FSSS_ND):

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

        assert self.mode == "test", "only support test mode!"
        test_object = cfg.DATASET.open_domain_test_object
        category_num_list = cfg.DATASET.open_domain_object_category_num

        LOGGER.info("current test class is {}".format(test_object))

        # initialize dir
        # for support sample
        init_dir = [os.path.join(self.source, i) for i in test_object]
        # specific defect category
        specific_defect = cfg.DATASET.open_domain_specific_defect_category
        self.specific_defect_dict = {i: specific_defect[idx] for idx, i in enumerate(test_object)}

        self.object_filename, self.object_category_filename, self.object_category_filename_list = \
            self.get_basic_element(init_dir, category_num_list)

        self.query_object_path = self.query_path_acquire(init_dir)  # list

    def __len__(self):
        return len(self.query_object_path)

    def __getitem__(self, idx):
        current_sample = self.query_object_path[idx]

        query_object, query_filename = current_sample.split("^")
        specific_defect = self.specific_defect_dict[query_object]

        sample_filename_list = self.object_category_filename[query_object][specific_defect]

        if self.shot >= len(sample_filename_list):
            acquire_k_shot_support = random.choices(sample_filename_list, k=self.shot)
        else:
            acquire_k_shot_support = random.sample(sample_filename_list, self.shot)

        support_img_path = [str(specific_defect)]+[i.replace("/", "_") for i in acquire_k_shot_support]

        self.random_crop_ratio = random.uniform(*self.crop_ratio)  # 0.7~1.2

        # for support set
        support_image_list = []
        support_mask_list = []
        support_defect_status = []

        for each_support_sample in acquire_k_shot_support:
            input_image, mask_defect, defect_status = self.support_mode_generate_image_mask(
                tmp_filename=each_support_sample,
                tmp_object=query_object,
                tmp_category=specific_defect,
                defect_generation_state=False,
                tmp_defect_mode=None
            )
            support_image_list.append(input_image)
            support_mask_list.append(mask_defect)
            support_defect_status.append(defect_status)

        support_image = torch.concat(support_image_list, dim=0)
        support_mask = torch.concat(support_mask_list, dim=0)
        support_defect_status_resize = sum(support_defect_status) == len(support_defect_status)

        sub_mode = "original"
        query_image, query_original_shape, query_crop_shape, query_input_shape, img_position_list = self.generate_image_mask_(
            tmp_filename=query_filename,
            tmp_object=query_object,
            support_defect_status_resize=support_defect_status_resize,
            sub_mode=sub_mode
        )

        """
        shape
        query_image: 3, image_longest_size, image_longest_size, torch.tensor, if train; else stack, 3, image_longest_size, image_longest_size
        query_original_shape: 2, tensor
        query_crop_shape: stack, 2 / 2 ratio > 0.1
        query_input_shape: stack, 2 / 2 ratio > 0.1
        img_pos: stack, position for point / [1] (train or ratio > 0.1) (top_h, down_h, left_w, right_w)
        support_image: k-shot, 3, image_longest_size, image_longest_size
        support_mask: k-shot, 1, image_longest_size, image_longest_size
        query_object_category: mIOU
        """
        return {
            "query_image": query_image,
            "query_original_shape": query_original_shape,
            "query_crop_shape": query_crop_shape,
            "query_input_shape": query_input_shape,
            "img_position_list": img_position_list,
            "support_image": support_image,
            "support_mask": support_mask,
            "query_object_category_filename": current_sample,
            "support_img_path": "_".join(support_img_path)
        }

    def generate_image_mask_(
            self,
            tmp_filename,
            tmp_object,
            support_defect_status_resize,
            sub_mode="scale"
    ):
        file_name = tmp_filename
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_image_h, original_image_w = img.shape[:2]
        original_img_shape = img.shape[:2]

        if support_defect_status_resize:
            # copy from sam lib
            # Transform the image to the form expected by the model
            input_pil_image = self.transform_original_image.image_convert_pilimage(img.astype(np.uint8))  # pil_image
            input_image_torch = self.transform_function(input_pil_image)
            input_image = self.preprocess(input_image_torch, self.image_longest_size)

            original_img_shape = img.shape[:2]
            crop_img_shape = img.shape[:2]
            img_position_list = [1]
            input_image_shape = tuple(input_image_torch.shape[-2:])
        else:
            crop_size = self.crop_size
            target_size_h, target_size_w = min(crop_size, original_image_h), min(crop_size, original_image_w)

            h_num = original_image_h // target_size_h
            w_num = original_image_w // target_size_w

            h_remainder = original_image_h % target_size_h
            w_remainder = original_image_w % target_size_w

            if h_remainder != 0:
                h_num += 1

            if w_remainder != 0:
                w_num += 1

            img_list = []
            crop_img_shape_list = []
            input_image_shape_list = []
            img_position_list = []  # (top_h, down_h, left_w, right_w)

            for temp_h in range(h_num):
                for temp_w in range(w_num):
                    if temp_h != h_num - 1 and temp_w != w_num - 1:
                        tmp_img = img[temp_h * target_size_h: (temp_h + 1) * target_size_h,
                                  temp_w * target_size_w: (temp_w + 1) * target_size_w, :]
                        img_position_list.append((temp_h * target_size_h, (temp_h + 1) * target_size_h,
                                                  temp_w * target_size_w, (temp_w + 1) * target_size_w))
                    elif temp_h == h_num - 1 and temp_w != w_num - 1:
                        tmp_img = img[original_image_h - target_size_h: original_image_h,
                                  temp_w * target_size_w: (temp_w + 1) * target_size_w, :]
                        img_position_list.append((original_image_h - target_size_h, original_image_h,
                                                  temp_w * target_size_w, (temp_w + 1) * target_size_w))
                    elif temp_w == w_num - 1 and temp_h != h_num - 1:
                        tmp_img = img[temp_h * target_size_h: (temp_h + 1) * target_size_h,
                                  original_image_w - target_size_w: original_image_w, :]
                        img_position_list.append((temp_h * target_size_h, (temp_h + 1) * target_size_h,
                                                  original_image_w - target_size_w, original_image_w))
                    else:
                        tmp_img = img[original_image_h - target_size_h: original_image_h,
                                  original_image_w - target_size_w: original_image_w, :]
                        img_position_list.append((original_image_h - target_size_h, original_image_h,
                                                  original_image_w - target_size_w, original_image_w))

                    input_pil_image = self.transform_original_image.image_convert_pilimage(
                        tmp_img.astype(np.uint8))  # pil_image
                    input_image_torch = self.transform_function(input_pil_image)
                    input_image = self.preprocess(input_image_torch, self.image_longest_size)

                    crop_img_shape_list.append(torch.as_tensor(tmp_img.shape[:2]))
                    input_image_shape_list.append(torch.as_tensor(tuple(input_image_torch.shape[-2:])))
                    img_list.append(input_image)

            # save bs(test=1), len_stack, 3, img_size, img_size
            input_image = torch.stack(img_list, dim=0)
            crop_img_shape = torch.stack(crop_img_shape_list, dim=0)
            input_image_shape = torch.stack(input_image_shape_list, dim=0)

        return input_image, torch.as_tensor(original_img_shape), torch.as_tensor(crop_img_shape), \
               torch.as_tensor(input_image_shape), img_position_list

    def get_basic_element(self, init_dir, category_num_list):
        object_category_filename = {}
        object_filename = {}
        object_category_filename_list = []

        for idx, tmp_path in enumerate(init_dir):
            filename_segmentation_category = {}

            tmp_object = tmp_path.split("/")[-1]
            support_path = os.path.join(tmp_path, "support", "image")
            mask_path = os.path.join(tmp_path, "support", "mask")

            image_path = glob(support_path+"/*.jpg")
            image_path += glob(support_path+"/*.png")

            for temp_image in image_path:
                image_save_name = "/".join(temp_image.split("/")[-3:])
                temp_dict = {}

                image_name = temp_image.split("/")[-1].split(".")[0]
                temp_mask = os.path.join(mask_path, image_name+"_mask.png")
                mask_label = np.array(PIL_Image.open(temp_mask))
                temp_dict["category_sum"] = 1
                temp_dict["seg"] = []
                temp_dict["category"] = []
                temp_dict["seg"].append(mask_label == 255)
                temp_dict["category"].append(0)

                filename_segmentation_category[image_save_name] = temp_dict

            object_filename[tmp_object] = filename_segmentation_category

            category_filename = generate_category_filename(category_num_list[idx],
                                                           filename_segmentation_category)

            object_category_filename[tmp_object] = category_filename

            for category, filename in category_filename.items():
                for tmp_filename in filename:
                    object_category_filename_list.append("^".join([tmp_object, str(category), tmp_filename]))

        return object_filename, object_category_filename, object_category_filename_list

    def query_path_acquire(self, init_dir):
        query_object_path = []

        for idx, tmp_path in enumerate(init_dir):
            tmp_object = tmp_path.split("/")[-1]
            query_path = os.path.join(tmp_path, "query", "image")

            image_path = glob(query_path + "/*.jpg")
            image_path += glob(query_path + "/*.png")

            for path in image_path:
                query_object_path.append("^".join([tmp_object, path]))

        return query_object_path
