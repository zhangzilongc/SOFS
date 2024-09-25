import numpy as np
import cv2
import os
import logging
import math
import torch
import torch.nn.functional as F
from typing import Union
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def freeze_paras(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output


# modify from monai dice loss
def dice_binary_loss(input: torch.Tensor,
                     target: torch.Tensor,
                     smooth_r: float = 1e5,
                     squared_pred=False,
                     reduction="mean") -> torch.Tensor:
    # inputs: b, h, w
    # target: b, h, w
    inputs_ = input
    beta = 1.
    smooth_nr_defect = 1e-5

    if target.shape != inputs_.shape:
        raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = torch.arange(1, len(inputs_.shape)).tolist()

    intersection = torch.sum(target * inputs_, dim=reduce_axis)

    if squared_pred:
        ground_o = torch.sum(target ** 2, dim=reduce_axis)
        pred_o = torch.sum(inputs_ ** 2, dim=reduce_axis)
    else:
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(inputs_, dim=reduce_axis)

    denominator = ground_o + pred_o
    ground_d_coefficient = (ground_o > 0.).float()
    ground_n_coefficient = (ground_o == 0.).float() * beta

    f_defect: torch.Tensor = 1.0 - (2.0 * intersection + smooth_nr_defect) / (denominator + smooth_nr_defect)
    f_normal: torch.Tensor = 1.0 - 1 / (smooth_r * pred_o + 1)

    f = f_defect * ground_d_coefficient + f_normal * ground_n_coefficient

    if reduction == "mean":
        # f = torch.sum(f) / (torch.sum(ground_o_coefficient) + smooth_nr)
        f = torch.mean(f)  # the batch and channel average
    elif reduction == "sum":
        f = torch.sum(f)  # sum over the batch and channel dims
    elif reduction == "none":
        # If we are not computing voxelwise loss components at least
        # make sure a none reduction maintains a broadcastable shape
        broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
        f = f.view(broadcast_shape)
    else:
        raise ValueError(f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].')

    return f


def dice_ce_loss_sum(y_m_squeeze, final_out, dice_weight, ce_weight, smooth_r=1e5):
    eps_ = 1e-6
    main_loss_ce = -torch.mean(
        y_m_squeeze * torch.log(final_out + eps_) + (1 - y_m_squeeze) * torch.log((1 - final_out) + eps_))
    main_loss_dice = dice_binary_loss(final_out, y_m_squeeze.float(), smooth_r=smooth_r)
    main_loss = dice_weight * main_loss_dice + ce_weight * main_loss_ce
    return main_loss


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)  # (bs, c * 9, h*w)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        ) # (bs, c, 3, 3, h*w)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3) # (bs, h*w, c, 3, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        # reshape(batchsize, -1)
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_bn(m):
    # BN
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def upsample_output_result(tmp_img, input_h, input_w, original_input_h, original_input_w, quantization=True):
    oav = tmp_img[:input_h, :input_w]
    oav = F.interpolate(oav.unsqueeze(0).unsqueeze(0).float(), size=(original_input_h, original_input_w),
                        mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    # quantization
    if quantization:
        oav = (oav > 0.9).float()
    return oav


def network_output2original_result(query_input_shape, query_original_shape, output_absolute_val, output_heatmap):
    # output_absolute_val, output_heatmap: h, w
    input_h, input_w = query_input_shape.numpy()
    input_h, input_w = int(input_h), int(input_w)

    original_input_h, original_input_w = query_original_shape.numpy()
    original_input_h, original_input_w = int(original_input_h), int(original_input_w)

    oav = upsample_output_result(
        tmp_img=output_absolute_val,
        input_h=input_h,
        input_w=input_w,
        original_input_h=original_input_h,
        original_input_w=original_input_w,
        quantization=True
    )

    ohm = upsample_output_result(
        tmp_img=output_heatmap,
        input_h=input_h,
        input_w=input_w,
        original_input_h=original_input_h,
        original_input_w=original_input_w,
        quantization=False
    )
    return oav, ohm


def acquire_training_miou(
        result_dict,
        query_object_category_filename,
        output_absolute_val,
        target
):
    for tmp_ocf, tmp_oav, tmp_mask in zip(query_object_category_filename,
                                          output_absolute_val,
                                          target):

        query_object, query_category, query_filename = tmp_ocf.split("^")
        if int(query_category) not in result_dict[query_object].keys():
            result_dict[query_object][int(query_category)] = {
                "intersection": [],
                "union": [],
                "new_target": []
            }

        intersection, union, target = intersectionAndUnionGPU(tmp_oav, tmp_mask.squeeze(0), 2, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        result_dict[query_object][int(query_category)]["intersection"].append(intersection)
        result_dict[query_object][int(query_category)]["union"].append(union)
        result_dict[query_object][int(query_category)]["new_target"].append(target)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)  # return shape [0的个数, 1的个数]
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def acquire_final_mIOU_FBIOU(result_dict, class_iou_class, class_miou, FB_IOU_intersection, FB_IOU_union):
    for query_object in result_dict.keys():
        object_category_dict = result_dict[query_object]
        for each_category in object_category_dict.keys():
            tmp_object_category = object_category_dict[each_category]
            tmp_intersection = np.array(tmp_object_category["intersection"])
            tmp_union = np.array(tmp_object_category["union"])

            tmp_intersection = np.sum(tmp_intersection, axis=0)
            tmp_union = np.sum(tmp_union, axis=0)

            tmp_fb_iou = tmp_intersection / tmp_union
            class_iou_class[query_object + "_" + str(each_category)] = {"background_iou": tmp_fb_iou[0],
                                                                        "foreground_iou": tmp_fb_iou[1]}
            class_miou += tmp_fb_iou[1]

            FB_IOU_intersection += tmp_intersection
            FB_IOU_union += tmp_union

    class_miou = class_miou * 1.0 / len(class_iou_class)
    FB_IOU = FB_IOU_intersection / FB_IOU_union

    return class_miou, class_iou_class, FB_IOU


def plot_qualitative_results(img, fig_save_path, file_name, rgba_img):
    plt.imshow(img)
    plt.imshow(rgba_img)
    plt.axis("off")
    plt.savefig(os.path.join(fig_save_path, file_name + "_on_mask.jpg"), bbox_inches='tight',
                pad_inches=0.0, dpi=300)
    plt.clf()


def plot_heatmap(img, ohm, fig_save_path, file_name):
    seg_each = np.clip(ohm * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(seg_each, cv2.COLORMAP_JET)

    heatmap_on_image = np.float32(heatmap) / 255 + np.float32(img) / 255
    heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
    heatmap_on_image = np.uint8(255 * heatmap_on_image)
    cv2.imwrite(os.path.join(fig_save_path, file_name + "_heatmap_on_ima.jpg"), heatmap_on_image)


def generate_rgba(original_mask, rgba):
    original_mask = original_mask.astype(int)

    original_mask_tensor = torch.tensor(original_mask)
    # 取固定通道的
    extra_channel = torch.gather(rgba, dim=0, index=original_mask_tensor.reshape(-1).unsqueeze(-1).repeat(1, 4))
    mask_rgba = extra_channel.reshape(original_mask.shape[0], original_mask.shape[1], 4).numpy()
    return mask_rgba


def produce_qualitative_result(
        original_mask,
        oav,
        ohm,
        source_path,
        query_object,
        query_filename,
        fig_save_path,
        test_num,
        support_img_path,
        moav=None,
        mohm=None
):
    # 1,0,0表示的是红色
    color_channel = torch.tensor(np.concatenate((np.array([[0, 0, 0]]), np.array([[1, 0, 0]])), axis=0))
    # 0.6表示的是透明度
    rgba = torch.tensor([[0.6]]).repeat(color_channel.shape[0], 1)
    rgba = torch.concat((color_channel, rgba), dim=1)
    rgba[0, -1] = 0

    oav = oav.cpu().numpy()  # h, w
    ohm = ohm.cpu().numpy()  # h, w

    original_mask = original_mask.cpu().numpy()
    original_img_path = os.path.join(source_path, query_object, query_filename)

    img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    true_rgba = generate_rgba(original_mask, rgba)
    oav_rgba = generate_rgba(oav, rgba)

    true_name = "_".join(
        [support_img_path, query_object, query_filename.split(".")[0].replace("/", "_"), str(test_num)])

    plt.imshow(img)
    plt.axis("off")
    plt.savefig(os.path.join(fig_save_path, true_name + "_image.jpg"), bbox_inches='tight',
                pad_inches=0.0, dpi=300)
    plt.clf()

    plot_qualitative_results(img=img, fig_save_path=fig_save_path, file_name=true_name + "true_label",
                             rgba_img=true_rgba)
    plot_qualitative_results(img=img, fig_save_path=fig_save_path, file_name=true_name + "output_label",
                             rgba_img=oav_rgba)

    plot_heatmap(img=img, fig_save_path=fig_save_path, file_name=true_name + "output", ohm=ohm)

    if moav is not None:
        moav = moav.cpu().numpy()
        mohm = mohm.cpu().numpy()

        moav_rgba = generate_rgba(moav, rgba)

        plot_qualitative_results(img=img, fig_save_path=fig_save_path, file_name=true_name + "meta_output_label",
                                 rgba_img=moav_rgba)

        plot_heatmap(img=img, fig_save_path=fig_save_path, file_name=true_name + "meta_output", ohm=mohm)


def produce_qualitative_result_open_domain(
        oav,
        ohm,
        query_object,
        query_filename,
        fig_save_path,
        test_num,
        support_img_path,
        moav=None,
        mohm=None
):
    # 1,0,0 denote red
    color_channel = torch.tensor(np.concatenate((np.array([[0, 0, 0]]), np.array([[1, 0, 0]])), axis=0))
    rgba = torch.tensor([[0.6]]).repeat(color_channel.shape[0], 1)
    rgba = torch.concat((color_channel, rgba), dim=1)
    rgba[0, -1] = 0

    oav = oav.cpu().numpy()  # h, w
    ohm = ohm.cpu().numpy()  # h, w
    original_img_path = query_filename

    img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    oav_rgba = generate_rgba(oav, rgba)

    true_name = "_".join(
        [support_img_path, query_object, query_filename.split("/")[-1].split(".")[0], str(test_num)])

    plt.imshow(img)
    plt.axis("off")
    plt.savefig(os.path.join(fig_save_path, true_name + "_image.jpg"), bbox_inches='tight',
                pad_inches=0.0, dpi=300)
    plt.clf()

    plot_qualitative_results(img=img, fig_save_path=fig_save_path, file_name=true_name + "output_label",
                             rgba_img=oav_rgba)

    plot_heatmap(img=img, fig_save_path=fig_save_path, file_name=true_name + "output", ohm=ohm)

    if moav is not None:
        moav = moav.cpu().numpy()
        mohm = mohm.cpu().numpy()

        moav_rgba = generate_rgba(moav, rgba)

        plot_qualitative_results(img=img, fig_save_path=fig_save_path, file_name=true_name + "meta_output_label",
                                 rgba_img=moav_rgba)

        plot_heatmap(img=img, fig_save_path=fig_save_path, file_name=true_name + "meta_output", ohm=mohm)
