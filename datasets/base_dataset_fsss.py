import os
import random
import torch
from torchvision import transforms
from PIL import ImageFilter
import numpy as np
import cv2

from datasets.utilis_data import ResizeLongestSide
from datasets import transform_tri
from torchvision.transforms.functional import to_pil_image
from torch.nn import functional as F
import logging
from glob import glob

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BASE_DATASET_FSSS(torch.utils.data.Dataset):
    """
    base dataset for few-shot semantic segmentation
    """

    def __init__(
            self,
            cfg,
            mode="train",
            **kwargs
    ):
        super().__init__()
        source = cfg.TRAIN.dataset_path if mode == 'train' else cfg.TEST.dataset_path
        self.source = source
        self.mode = mode
        self.data_split = "split_" + str(cfg.DATASET.split)
        self.image_longest_size = cfg.DATASET.image_size  # max(height, width) of original image
        self.mask_longest_size = cfg.DATASET.mask_size  # max(height, width) of original mask
        self.test_unified_mask_longest_size = cfg.DATASET.unified_mask_size
        self.test_sample_repeated_multiple = cfg.DATASET.test_sample_repeated_multiple
        self.shot = cfg.DATASET.shot
        self.transform_original_image = ResizeLongestSide(self.image_longest_size)
        self.transform_mask = ResizeLongestSide(self.mask_longest_size)

        self.first_step_transform_train = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3)
        ])
        mean = [0.485, 0.456, 0.406]
        mean = [item * 255 for item in mean]

        self.second_step_transform_train = transform_tri.Compose([
            transform_tri.RandRotate([cfg.DATASET.rotate_min, cfg.DATASET.rotate_max], padding=mean, ignore_label=0),
            transform_tri.RandomGaussianBlur(p=0.3),
            transform_tri.RandomHorizontalFlip()])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        self.transform_function = transform_test

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

        self.object_filename = {}
        self.object_category_filename = {}
        self.object_category_filename_list = {}

    def __len__(self):
        if self.mode == "train":
            return len(self.object_category_filename_list)
        else:
            return len(self.object_category_filename_list) * self.test_sample_repeated_multiple

    def __getitem__(self, idx):
        if self.mode == "train":
            tmp_idx = idx
        else:
            tmp_idx = idx // self.test_sample_repeated_multiple

        current_sample = self.object_category_filename_list[tmp_idx]

        query_object, query_category, query_filename = current_sample.split("^")
        query_category = int(query_category)

        sample_filename_list = self.object_category_filename[query_object][query_category]

        acquire_k_shot_support = [query_filename]
        while query_filename in acquire_k_shot_support:
            if self.shot > len(sample_filename_list):
                acquire_k_shot_support = random.choices(sample_filename_list, k=self.shot)
            else:
                acquire_k_shot_support = random.sample(sample_filename_list, self.shot)

        support_img_path = [str(query_category)] + [i.replace("/", "_") for i in acquire_k_shot_support]

        # for support set
        support_image_list = []
        support_mask_list = []

        for each_support_sample in acquire_k_shot_support:
            input_image, mask_defect, original_img_shape, input_image_shape = self.generate_image_mask(
                tmp_filename=each_support_sample,
                tmp_object=query_object,
                tmp_category=query_category
            )
            support_image_list.append(input_image)
            support_mask_list.append(mask_defect)

        support_image = torch.stack(support_image_list, dim=0)
        support_mask = torch.stack(support_mask_list, dim=0)

        sub_mode = "scale" if self.mode == "train" else "original"
        # only for test
        query_image, query_mask, query_original_shape, query_input_shape = self.generate_image_mask(
            tmp_filename=query_filename,
            tmp_object=query_object,
            tmp_category=query_category,
            sub_mode=sub_mode
        )

        """
        shape
        query_image: 3, image_longest_size, image_longest_size, torch.tensor
        query_mask: 1, mask_longest_size, mask_longest_size, torch.tensor
        query_original_shape: 2, tensor
        support_image: k-shot, 3, image_longest_size, image_longest_size
        support_mask: k-shot, 1, image_longest_size, image_longest_size
        query_object_category: mIOU
        """
        return {
            "query_image": query_image,
            "query_mask": query_mask,
            "query_original_shape": query_original_shape,
            "query_input_shape": query_input_shape,
            "support_image": support_image,
            "support_mask": support_mask,
            "query_object_category_filename": current_sample,
            "support_img_path": "_".join(support_img_path)
        }

    def generate_image_mask(self, tmp_filename, tmp_object, tmp_category, sub_mode="scale"):
        file_name = tmp_filename
        filename_segmentation_category = self.object_filename[tmp_object]

        attribute = filename_segmentation_category[file_name]
        category_list = attribute["category"]
        category_idx_pointer = category_list.index(tmp_category)

        temp_mask = attribute["seg"][category_idx_pointer]
        temp_mask = (temp_mask * 255).astype(np.uint8)

        path_now = os.path.join(self.source, tmp_object, file_name)

        img = cv2.imread(path_now, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            img = np.array(self.first_step_transform_train(to_pil_image(img)))
            img, temp_mask = self.second_step_transform_train(img, temp_mask)
            temp_mask = temp_mask.astype(np.uint8)

        # copy from sam lib
        # Transform the image to the form expected by the model

        input_pil_image = self.transform_original_image.image_convert_pilimage(img.astype(np.uint8))  # pil_image
        input_image_torch = self.transform_function(input_pil_image)
        input_image = self.preprocess(input_image_torch, self.image_longest_size)

        original_img_shape = img.shape[:2]
        input_image_shape = tuple(input_image_torch.shape[-2:])

        if sub_mode == "scale":
            current_mask_transform = self.transform_mask.apply_image(temp_mask)
            current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])

            mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")
        else:
            current_mask_torch = torch.as_tensor(temp_mask[None, :, :])
            mask_defect = self.preprocess(current_mask_torch, self.test_unified_mask_longest_size, mode="gray")
        mask_defect = (mask_defect > 0.1).float()

        return input_image, mask_defect, torch.as_tensor(original_img_shape), torch.as_tensor(input_image_shape)

    # modified from sam
    def preprocess(self, x: torch.Tensor, imagesize, mode="rgb") -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        if mode == "gray":
            mean = 0.
            std = torch.max(x)
            x = (x - mean) / std

        h, w = x.shape[-2:]
        padh = imagesize - h
        padw = imagesize - w
        # numpy is (last_second_h, last_second_w, last_one_h, last_one_w)
        # torch is (last_one_h, last_one_w, last_second_h, last_second_w)
        x = F.pad(x, (0, padw, 0, padh))
        return x


# data transform
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
