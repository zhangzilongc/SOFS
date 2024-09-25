from utils.multiprocessing import is_master_proc
import logging
import time
from utils.common import AverageMeter, intersectionAndUnionGPU, seed_everything, \
    acquire_final_mIOU_FBIOU, produce_qualitative_result_open_domain, upsample_output_result, acquire_training_miou,\
    network_output2original_result, fix_bn
import numpy as np
import torch
import os
import cv2
from datasets.base_dataset_fsss import IMAGENET_MEAN, IMAGENET_STD

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def opendomain_eval(val_loader, model, epoch, cfg, rand_seed):
    LOGGER.info("Start validate!")
    model_time = AverageMeter()

    if torch.distributed.is_initialized():
        temp_rank = torch.distributed.get_rank()
    else:
        temp_rank = 0
    seed_everything(rand_seed + temp_rank)
    current_method = cfg.TEST.method

    model.eval()
    end = time.time()
    val_start = end

    """
    result_dict: 
    {
    object:
        {
            category: 
            {
            intersection: [每个样本（包括同一个query不同的support对应的）的值] 
            union: []
            new_target: []
            }
        }
    }
    """

    test_num = 0

    for i, data in enumerate(val_loader):
        s_input = data["support_image"]
        s_mask = data["support_mask"]
        input = data["query_image"]
        assert input.dim() in [4, 5], "must statisfy the predefined dim"
        grid_size = input.shape[1]
        assert input.shape[0] == 1, "do not support bs > 1!"
        query_original_shape = data["query_original_shape"][0]
        query_input_shape = data["query_input_shape"][0]
        query_crop_shape = data["query_crop_shape"][0]
        img_position_list = data["img_position_list"]
        query_object_category_filename = data["query_object_category_filename"][0]
        support_img_path = data["support_img_path"][0]

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        if input.dim() == 4:
            start_time = time.time()
            if current_method in ["SOFS"]:
                output = model(s_x=s_input, s_y=s_mask, x=input)
            elif current_method in ["SegGPT"]:
                output = SegGPT_validate(input=input, s_input=s_input, s_mask=s_mask, model=model)
            else:
                raise NotImplementedError

            model_time.update(time.time() - start_time)
            if current_method in ["SOFS"]:
                output_absolute_val = output.max(1)[1][0]
                # heatmap
                output_heatmap = output[:, 1, ...][0]
            elif current_method in ["SegGPT"]:
                output_absolute_val = (output > cfg.TEST.semantic_threshold).to(int)
                # heatmap
                output_heatmap = output

            original_output, original_heatmap = network_output2original_result(
                query_input_shape=query_input_shape,
                query_original_shape=query_original_shape,
                output_absolute_val=output_absolute_val,
                output_heatmap=output_heatmap
            )
        else:
            original_output = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_heatmap = torch.zeros(
                (grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_position = torch.zeros(
                (grid_size, int(query_original_shape[0]), int(query_original_shape[1])))

            multiple_ND = grid_size // cfg.TEST_SETUPS.ND_batch_size
            multiple_mod = grid_size % cfg.TEST_SETUPS.ND_batch_size
            multiple_bs = cfg.TEST_SETUPS.ND_batch_size
            iter_num = multiple_ND if multiple_mod == 0 else multiple_ND+1

            start_time = time.time()
            if current_method in ["SOFS"]:
                output_total = []
                for pointer_idx in range(iter_num):
                    init_val = pointer_idx*cfg.TEST_SETUPS.ND_batch_size
                    _, gs, c, h, w = input.shape
                    if multiple_ND == 0:
                        multiple_bs = gs

                    if pointer_idx == multiple_ND:
                        multiple_bs = multiple_mod

                    support_input_m = s_input.repeat(multiple_bs, 1, 1, 1, 1)
                    support_mask_m = s_mask.repeat(multiple_bs, 1, 1, 1, 1)
                    query_input_m = input[:, init_val: init_val+multiple_bs, ...].reshape(1*multiple_bs, c, h, w)

                    tmp_output = model(s_x=support_input_m, s_y=support_mask_m, x=query_input_m)
                    output_total.append(tmp_output)

                output_total = torch.concat(output_total, dim=0)

            for query_idx in range(grid_size):
                current_position = img_position_list[query_idx]
                current_position = [int(i) for i in current_position]
                top, down, left, right = current_position

                if current_method in ["SOFS"]:
                    # output = model(s_x=s_input, s_y=s_mask, x=input[:, query_idx, ...])
                    output = output_total[query_idx].unsqueeze(0)
                elif current_method in ["SegGPT"]:
                    output = SegGPT_validate(input=input[:, query_idx, ...], s_input=s_input, s_mask=s_mask,
                                             model=model)
                else:
                    raise NotImplementedError

                if current_method in ["SOFS"]:
                    output_absolute_val = output.max(1)[1][0]
                    # heatmap
                    output_heatmap = output[:, 1, ...][0]
                elif current_method in ["SegGPT"]:
                    output_absolute_val = (output > cfg.TEST.semantic_threshold).to(int)
                    # heatmap
                    output_heatmap = output

                tuple_output_absolute_val_shape = tuple(torch.as_tensor(output_absolute_val.shape).numpy())
                tuple_query_input_shape = tuple(query_input_shape[query_idx].numpy())
                tuple_query_crop_shape = tuple(query_crop_shape[query_idx].numpy())

                if tuple_output_absolute_val_shape == tuple_query_input_shape and tuple_query_input_shape == tuple_query_crop_shape:
                    # print("pass")
                    pass
                else:
                    # print(1)
                    output_absolute_val, output_heatmap = network_output2original_result(
                        query_input_shape=query_input_shape[query_idx],
                        query_original_shape=query_crop_shape[query_idx],
                        output_absolute_val=output_absolute_val,
                        output_heatmap=output_heatmap
                    )

                original_output[query_idx, top: down, left: right] = output_absolute_val.cpu()
                original_heatmap[query_idx, top: down, left: right] = output_heatmap.cpu()
                original_position[query_idx, top: down, left: right] = 1

            model_time.update(time.time() - start_time)
            # return h, w
            original_output = torch.sum(original_output, dim=0) / torch.sum(original_position, dim=0)
            original_heatmap = torch.sum(original_heatmap, dim=0) / torch.sum(original_position, dim=0)

        query_object, query_filename = query_object_category_filename.split("^")

        if cfg.TEST.VISUALIZE.save_figure:
            test_num += 1
            fig_save_path = os.path.join(cfg.OUTPUT_DIR, "figure_save")
            os.makedirs(fig_save_path, exist_ok=True)

            try:
                produce_qualitative_result_open_domain(
                    oav=original_output,
                    ohm=original_heatmap,
                    query_object=query_object,
                    query_filename=query_filename,
                    fig_save_path=fig_save_path,
                    test_num=test_num,
                    support_img_path=support_img_path
                )
            except:
                pass

    val_time = time.time() - val_start


def SegGPT_validate(
        input,
        s_input,
        s_mask,
        model
):
    assert input.shape[0] == 1, "do not support bs > 1!"
    input = input.repeat(s_input.shape[1], 1, 1, 1)
    seggpt_input = torch.concat((s_input.squeeze(0), input), dim=2)

    seggpt_mask = s_mask.squeeze(0).repeat(1, 3, 1, 1)
    seggpt_mask = seggpt_mask - torch.as_tensor(IMAGENET_MEAN).reshape(1, -1, 1, 1).cuda()
    seggpt_mask = seggpt_mask / torch.as_tensor(IMAGENET_STD).reshape(1, -1, 1, 1).cuda()
    seggpt_mask = torch.concat((seggpt_mask, seggpt_mask), dim=2)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(seggpt_mask)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])

    feat_ensemble = 0 if len(seggpt_input) > 1 else -1
    _, y, mask = model(seggpt_input.float(), seggpt_mask.float(), bool_masked_pos.cuda(),
                       valid.float().cuda(), seg_type.cuda(), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y)

    output = y[0, y.shape[1] // 2:, :, :]
    output = torch.clip((output * torch.as_tensor(IMAGENET_STD).cuda()
                         + torch.as_tensor(IMAGENET_MEAN).cuda()) * 255, 0, 255).permute(2, 0, 1)
    output = torch.max(output, dim=0)[0] / 255
    return output