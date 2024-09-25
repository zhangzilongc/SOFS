import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple
import cv2
import json


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def image_convert_pilimage(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return resize(to_pil_image(image), target_size)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        # (y height, x width)
        return (newh, neww)


def obtain_filename_segmentation_category_dict(path):
    train_val = path.split("/")[-1]

    annotation_path = os.path.join(path, "_annotations.coco.json")

    with open(annotation_path) as f:
        json_data = json.load(f)

    # 获取类别数目，文件名字以及每个文件的属性
    category_num = len(json_data["categories"])
    id_file_name = {i['id']: i['file_name'] for i in json_data["images"]}

    attribute_dict = {}
    for i in json_data["annotations"]:
        if i["image_id"] not in attribute_dict.keys():
            attribute_dict[i["image_id"]] = []

        attribute_dict[i["image_id"]].append(i)

    # scan json convert to segmentation (polygon)
    filename_segmentation_category = {}

    for id_key in id_file_name.keys():
        filename = id_file_name[id_key]
        filename = "/".join([train_val, filename])
        filename_segmentation_category[filename] = {"seg": [],
                                                    "category": []}

        image = cv2.imread(os.path.join(path, id_file_name[id_key]))
        h, w = image.shape[:2]

        temp_seg = []
        temp_cate = []

        count_mask = 0
        for each_attribute in attribute_dict[id_key]:
            mask = each_attribute["segmentation"]
            category_id = each_attribute['category_id']

            category_mask = np.zeros((h, w), dtype=np.int32)
            for each_mask in mask:
                temp_mask = np.zeros((h, w), dtype=np.int32)
                obj = np.array(each_mask, dtype=np.int32).reshape(1, len(each_mask) // 2, 2)
                cv2.fillPoly(temp_mask, obj, 1)

                category_mask += temp_mask
                count_mask += 1

            temp_seg.append(category_mask > 0)
            temp_cate.append(category_id)

        cate_set = list(set(temp_cate))

        # 保证相同的类别的分割相融合
        for i in cate_set:
            index_ = get_index1(temp_cate, i)

            category_mask = np.zeros((h, w), dtype=np.int32)
            for seg_index in index_:
                category_mask += temp_seg[seg_index]

            # 保证类别和seg相对应
            filename_segmentation_category[filename]["seg"].append(category_mask > 0)
            filename_segmentation_category[filename]["category"].append(i)

        filename_segmentation_category[filename]["category_sum"] = len(set(temp_cate))
        filename_segmentation_category[filename]["seglen_sum"] = count_mask

    return filename_segmentation_category, category_num


def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]


def generate_category_filename(category_num, filename_segmentation_category):
    category_filename = {}

    for i in range(category_num):
        category_filename[i] = []

    for key, val in filename_segmentation_category.items():
        for tmp_cate in val['category']:
            category_filename[tmp_cate].append(key)
    return category_filename

