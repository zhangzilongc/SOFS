import os
import random
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from datasets.base_dataset_fsss import BASE_DATASET_FSSS
import logging

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BASE_DATASET_FSSS_ND(BASE_DATASET_FSSS):

    def __init__(
            self,
            cfg,
            mode="train",
            **kwargs
    ):
        super().__init__(cfg=cfg, mode=mode, kwargs=kwargs)
        self.area_resize_ratio = cfg.DATASET.area_resize_ratio
        self.crop_size = cfg.DATASET.crop_size
        self.crop_ratio = cfg.DATASET.crop_ratio
        # only for semantic segmentation, sample multiple crop in one image
        self.s_in_shot = cfg.DATASET.s_in_shot
        self.rand_seed = cfg.RNG_SEED
        self.normal_sample_sampling_prob = cfg.DATASET.normal_sample_sampling_prob

    def __getitem__(self, idx):
        if self.mode == "train":
            tmp_idx = idx
        else:
            tmp_idx = idx // self.test_sample_repeated_multiple

        current_sample = self.object_category_filename_list[tmp_idx]

        query_object, query_category, query_filename = current_sample.split("^")
        query_category = int(query_category)

        support_category = query_category

        sample_filename_list = self.object_category_filename[query_object][support_category]

        acquire_k_shot_support = [query_filename]
        while query_filename in acquire_k_shot_support:
            if self.shot > len(sample_filename_list):
                acquire_k_shot_support = random.choices(sample_filename_list, k=self.shot)
            else:
                acquire_k_shot_support = random.sample(sample_filename_list, self.shot)

        support_img_path = [str(support_category)]+[i.replace("/", "_") for i in acquire_k_shot_support]

        generate_defect_state = False
        tmp_defect_mode = None

        # for support set
        support_image_list = []
        support_mask_list = []
        # store the area ratio of each support defect
        support_defect_status = []

        for each_support_sample in acquire_k_shot_support:
            input_image, mask_defect, defect_status = self.support_mode_generate_image_mask(
                tmp_filename=each_support_sample,
                tmp_object=query_object,
                tmp_category=support_category,
                defect_generation_state=generate_defect_state,
                tmp_defect_mode=tmp_defect_mode
            )
            support_image_list.append(input_image)
            support_mask_list.append(mask_defect)
            support_defect_status.append(defect_status)

        support_image = torch.concat(support_image_list, dim=0)
        support_mask = torch.concat(support_mask_list, dim=0)
        support_defect_status_resize = sum(support_defect_status) == len(support_defect_status)

        sub_mode = "scale" if self.mode == "train" else "original"
        query_image, query_mask, query_original_shape, query_crop_shape, query_input_shape, img_position_list = self.generate_image_mask_(
            tmp_filename=query_filename,
            tmp_object=query_object,
            tmp_category=query_category,
            support_defect_status_resize=support_defect_status_resize,
            sub_mode=sub_mode,
            defect_generation_state=generate_defect_state,
            tmp_defect_mode=tmp_defect_mode
        )

        """
        shape
        query_image: 3, image_longest_size, image_longest_size, torch.tensor, if train; else stack, 3, image_longest_size, image_longest_size
        query_mask: 1, mask_longest_size, mask_longest_size, torch.tensor
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
            "query_mask": query_mask,
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
            tmp_category,
            support_defect_status_resize,
            defect_generation_state,
            tmp_defect_mode,
            sub_mode="scale"
    ):
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

        original_image_h, original_image_w = img.shape[:2]
        original_img_shape = img.shape[:2]

        processed_mask = np.array((temp_mask / 255) > 0).astype(np.uint8)
        # only for train
        defect_area_ratio = np.sum(processed_mask) / (temp_mask.shape[0] * temp_mask.shape[1])

        if (defect_area_ratio > self.area_resize_ratio and self.mode == "train") or (support_defect_status_resize and self.mode != "train"):
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
            crop_img_shape = img.shape[:2]
            img_position_list = [1]
            input_image_shape = tuple(input_image_torch.shape[-2:])

            if sub_mode == "scale":
                current_mask_transform = self.transform_mask.apply_image(temp_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])

                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")
            else:
                current_mask_torch = torch.as_tensor(temp_mask[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.test_unified_mask_longest_size,
                                              mode="gray")
        else:
            if self.mode == "train":
                random_crop_ratio = random.uniform(*self.crop_ratio)  # 0.7~1.2
                crop_size = int(self.crop_size * random_crop_ratio)
            else:
                crop_size = self.crop_size
            target_size_h, target_size_w = min(crop_size, original_image_h), min(crop_size, original_image_w)

            if sub_mode == "scale":
                # crop
                if random.uniform(0, 1) > self.normal_sample_sampling_prob:
                    condition_mask_h, condition_mask_w = np.where(processed_mask > 0)
                else:
                    condition_mask_h, condition_mask_w = np.where(processed_mask == 0)
                len_processed_mask = len(condition_mask_h)
                center_pixel_idx = random.randint(0, len_processed_mask - 1)

                center_pixel = (condition_mask_h[center_pixel_idx], condition_mask_w[center_pixel_idx])
                residual_h, residual_w = original_image_h - center_pixel[0], original_image_w - center_pixel[1]

                mask_down_boundary_random, mask_right_boundary_random = random.randint(0, crop_size), random.randint(0,
                                                                                                                     crop_size)
                real_mask_down = min(residual_h, mask_down_boundary_random)
                real_mask_right = min(residual_w, mask_right_boundary_random)

                down_boundary = center_pixel[0] + real_mask_down
                right_boundary = center_pixel[1] + real_mask_right
                top_boundary = down_boundary - target_size_h
                left_boundary = right_boundary - target_size_w

                if top_boundary < 0:
                    top_boundary = 0
                    down_boundary = target_size_h

                if left_boundary < 0:
                    left_boundary = 0
                    right_boundary = target_size_w

                crop_img = img[top_boundary: down_boundary, left_boundary: right_boundary, :]
                crop_mask = temp_mask[top_boundary: down_boundary, left_boundary: right_boundary]

                # 第一，二步只对训练模式起作用
                if self.mode == "train":

                    crop_img = np.array(self.first_step_transform_train(to_pil_image(crop_img)))
                    crop_img, crop_mask = self.second_step_transform_train(crop_img, crop_mask)
                    crop_mask = crop_mask.astype(np.uint8)

                # copy from sam lib
                # Transform the image to the form expected by the model
                input_pil_image = self.transform_original_image.image_convert_pilimage(
                    crop_img.astype(np.uint8))  # pil_image
                input_image_torch = self.transform_function(input_pil_image)
                input_image = self.preprocess(input_image_torch, self.image_longest_size)

                current_mask_transform = self.transform_mask.apply_image(crop_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])

                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")

                crop_img_shape = crop_img.shape[:2]
                input_image_shape = tuple(input_image_torch.shape[-2:])
                img_position_list = [1]
            else:
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

                current_mask_torch = torch.as_tensor(temp_mask[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.test_unified_mask_longest_size,
                                              mode="gray")  # 1, 256, 256
        mask_defect = (mask_defect > 0.1).float()

        return input_image, mask_defect, torch.as_tensor(original_img_shape), torch.as_tensor(crop_img_shape), \
               torch.as_tensor(input_image_shape), img_position_list

    def support_mode_generate_image_mask(
            self,
            tmp_filename,
            tmp_object,
            tmp_category,
            defect_generation_state,
            tmp_defect_mode
    ):
        file_name = tmp_filename
        filename_segmentation_category = self.object_filename[tmp_object]

        attribute = filename_segmentation_category[file_name]
        category_list = attribute["category"]
        category_idx_pointer = category_list.index(tmp_category)

        temp_mask = attribute["seg"][category_idx_pointer]
        temp_mask = (temp_mask * 255).astype(np.uint8)

        # join can not append /
        path_now = os.path.join(self.source, tmp_object, file_name)

        img = cv2.imread(path_now, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_image_h, original_image_w = img.shape[:2]

        processed_mask = np.array((temp_mask / 255) > 0).astype(np.uint8)
        defect_area_ratio = np.sum(processed_mask) / (temp_mask.shape[0] * temp_mask.shape[1])

        defect_status = defect_area_ratio > self.area_resize_ratio

        if defect_area_ratio > self.area_resize_ratio:
            if self.mode == "train":

                img = np.array(self.first_step_transform_train(to_pil_image(img)))
                img, temp_mask = self.second_step_transform_train(img, temp_mask)
                temp_mask = temp_mask.astype(np.uint8)

            # copy from sam lib
            # Transform the image to the form expected by the model

            input_pil_image = self.transform_original_image.image_convert_pilimage(img.astype(np.uint8))  # pil_image
            input_image_torch = self.transform_function(input_pil_image)
            input_image = self.preprocess(input_image_torch, self.image_longest_size)
            input_image = input_image.unsqueeze(0).repeat(self.s_in_shot, 1, 1, 1)

            current_mask_transform = self.transform_mask.apply_image(temp_mask)  # (256, min_len)
            current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])

            mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")  # 1, 256, 256
            mask_defect = mask_defect.unsqueeze(0).repeat(self.s_in_shot, 1, 1, 1)
        else:
            # crop
            condition_mask_h, condition_mask_w = np.where(processed_mask > 0)
            len_processed_mask = len(condition_mask_h)

            s_in_shot_img_list = []
            s_in_shot_mask_list = []
            for _ in range(self.s_in_shot):
                if self.mode == "train":
                    # scale invariant
                    random_crop_ratio = random.uniform(*self.crop_ratio)  # 0.7~1.2
                    crop_size = int(self.crop_size * random_crop_ratio)
                else:
                    crop_size = self.crop_size
                target_size_h, target_size_w = min(crop_size, original_image_h), min(crop_size, original_image_w)

                center_pixel_idx = random.randint(0, len_processed_mask - 1)

                center_pixel = (condition_mask_h[center_pixel_idx], condition_mask_w[center_pixel_idx])
                residual_h, residual_w = original_image_h - center_pixel[0], original_image_w - center_pixel[1]

                mask_down_boundary_random, mask_right_boundary_random = random.randint(0, crop_size), random.randint(0,
                                                                                                                     crop_size)
                real_mask_down = min(residual_h, mask_down_boundary_random)
                real_mask_right = min(residual_w, mask_right_boundary_random)

                down_boundary = center_pixel[0] + real_mask_down
                right_boundary = center_pixel[1] + real_mask_right
                top_boundary = down_boundary - target_size_h
                left_boundary = right_boundary - target_size_w

                if top_boundary < 0:
                    top_boundary = 0
                    down_boundary = target_size_h

                if left_boundary < 0:
                    left_boundary = 0
                    right_boundary = target_size_w

                crop_img = img[top_boundary: down_boundary, left_boundary: right_boundary, :]
                crop_mask = temp_mask[top_boundary: down_boundary, left_boundary: right_boundary]

                if self.mode == "train":

                    crop_img = np.array(self.first_step_transform_train(to_pil_image(crop_img)))
                    crop_img, crop_mask = self.second_step_transform_train(crop_img, crop_mask)
                    crop_mask = crop_mask.astype(np.uint8)

                # copy from sam lib
                # Transform the image to the form expected by the model

                input_pil_image = self.transform_original_image.image_convert_pilimage(
                    crop_img.astype(np.uint8))  # pil_image
                input_image_torch = self.transform_function(input_pil_image)
                input_image = self.preprocess(input_image_torch, self.image_longest_size)

                current_mask_transform = self.transform_mask.apply_image(crop_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])

                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")

                s_in_shot_img_list.append(input_image)
                s_in_shot_mask_list.append(mask_defect)

            input_image = torch.stack(s_in_shot_img_list, dim=0)
            mask_defect = torch.stack(s_in_shot_mask_list, dim=0)

        mask_defect = (mask_defect > 0.1).float()

        return input_image, mask_defect, defect_status
