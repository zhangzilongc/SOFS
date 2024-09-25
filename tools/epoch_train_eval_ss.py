from utils.multiprocessing import is_master_proc
import logging
import time
from utils.common import AverageMeter, intersectionAndUnionGPU, seed_everything, \
    acquire_final_mIOU_FBIOU, produce_qualitative_result, upsample_output_result, acquire_training_miou,\
    network_output2original_result, fix_bn
from tools.optimizer_schedule import poly_learning_rate
import numpy as np
import torch
import os
import cv2
from datasets.base_dataset_fsss import IMAGENET_MEAN, IMAGENET_STD
from datasets.dataset_split import VISION_V1_split_train_dict, VISION_V1_split_test_dict, VISION_V1_split_train_val_dict, \
    DS_Spectrum_DS_split_test_dict

LOGGER = logging.getLogger(__name__)


def epoch_train_ss(train_loader, model, optimizer, epoch, cfg, validate_each_class=False):
    if torch.distributed.is_initialized():
        temp_rank = torch.distributed.get_rank()
    else:
        temp_rank = 0
    # Set random seed from configs.
    seed_everything(cfg.RNG_SEED + temp_rank + epoch)

    current_method = cfg.TRAIN.method

    data_split = "split_" + str(cfg.DATASET.split)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if current_method in ["SOFS"]:
        model.apply(fix_bn)
    if hasattr(model.module, 'backbone'):
        model.module.backbone.eval()
    if epoch == 1:
        if is_master_proc():
            LOGGER.info("backbone eval mode, model train")

    end = time.time()
    val_time = 0.
    max_iter = cfg.TRAIN_SETUPS.epochs * len(train_loader)
    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND"]:
        result_dict = {i: {} for i in VISION_V1_split_train_dict[data_split]}
    else:
        raise NotImplementedError

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1

        if cfg.TRAIN_SETUPS.poly_training:
            if current_method in ["SOFS"]:
                if cfg.TRAIN_SETUPS.learning_rate > 1e-6:
                    index_split_ = 0

                    poly_learning_rate(
                        optimizer,
                        cfg.TRAIN_SETUPS.learning_rate,
                        current_iter,
                        max_iter,
                        power=0.9,
                        index_split=index_split_,
                        warmup=False,
                        warmup_step=len(train_loader) // 2,
                        scale_lr=cfg.TRAIN_SETUPS.lr_multiple
                    )

        s_input = data["support_image"]
        s_mask = data["support_mask"]
        input = data["query_image"]
        target = data["query_mask"]
        query_object_category_filename = data["query_object_category_filename"]

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if current_method in ["SOFS"]:
            output, main_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            loss = main_loss
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = input.size(0)  # batch_size

        if validate_each_class:
            acquire_training_miou(
                result_dict=result_dict,
                query_object_category_filename=query_object_category_filename,
                output_absolute_val=output,
                target=target
            )

        intersection, union, target = intersectionAndUnionGPU(output, target.squeeze(1), 2, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        main_loss_meter.update(main_loss.item(), n)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # recall
    recall_class = intersection_meter.sum / (target_meter.sum + 1e-10)  # recall
    FB_IOU = np.mean(iou_class)
    mRecall = np.mean(recall_class)

    if is_master_proc():
        if current_method in ["SOFS"]:
            LOGGER.info(
                'Train result at epoch [{}/{}]: data_time: {:.2f}, batch_time: {:.2f} loss: {:.4f}, main_loss: {:.4f}, FB_IOU/mRecall {:.4f}/{:.4f}.'.format(
                    epoch, cfg.TRAIN_SETUPS.epochs,
                    data_time.avg, batch_time.avg,
                    loss_meter.avg, main_loss_meter.avg,
                    FB_IOU, mRecall))

        for i in range(2):
            LOGGER.info('Class_{} Result: FB_IOU/Recall {:.4f}/{:.4f}.'.format(i, iou_class[i], recall_class[i]))

    if validate_each_class:
        if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND"]:
            result_dict_total = {i: {} for i in VISION_V1_split_train_dict[data_split]}
        else:
            raise NotImplementedError

        result_dict_gather = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(result_dict_gather, result_dict)

        if is_master_proc():
            for each_dict in result_dict_gather:
                for each_object, category_attribute_val in each_dict.items():
                    for category_, attribute_val in category_attribute_val.items():
                        if category_ not in result_dict_total[each_object].keys():
                            result_dict_total[each_object][category_] = {
                                "intersection": [],
                                "union": [],
                                "new_target": []
                            }

                        for attribute_, val_ in attribute_val.items():
                            result_dict_total[each_object][category_][attribute_].extend(val_)

            class_iou_class = {}
            class_miou = 0
            FB_IOU_intersection = np.array([0., 0.])
            FB_IOU_union = np.array([0., 0.])

            class_miou, class_iou_class, FB_IOU = acquire_final_mIOU_FBIOU(
                result_dict=result_dict_total,
                class_iou_class=class_iou_class,
                class_miou=class_miou,
                FB_IOU_intersection=FB_IOU_intersection,
                FB_IOU_union=FB_IOU_union
            )

            if is_master_proc():
                LOGGER.info("current epoch is {}".format(epoch))
                LOGGER.info('meanIoU---Val result: mIoU_final {:.4f}.'.format(class_miou))

                LOGGER.info('<<<<<<< Every Class Results <<<<<<<')
                for i in class_iou_class.keys():
                    LOGGER.info('{} Foreground Result: iou_f {:.4f}.'.format(i, class_iou_class[i]["foreground_iou"]))
                    LOGGER.info('{} Background Result: iou_b {:.4f}.'.format(i, class_iou_class[i]["background_iou"]))
                for i in range(2):
                    LOGGER.info('Class_{} Result: FB_IOU {:.4f}.'.format(i, FB_IOU[i]))

                LOGGER.info('FBIoU---Val result: FBIoU {:.4f}.'.format(np.mean(FB_IOU)))
                LOGGER.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@torch.no_grad()
def epoch_validate_ss(val_loader, model, epoch, cfg, rand_seed, train_validate=False):
    LOGGER.info("Start validate!")
    model_time = AverageMeter()

    data_split = "split_" + str(cfg.DATASET.split)

    if torch.distributed.is_initialized():
        temp_rank = torch.distributed.get_rank()
    else:
        temp_rank = 0
    # Set random seed from configs.
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
            intersection: [] 
            union: []
            new_target: []
            }
        }
    }
    """

    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND"]:
        result_dict = {i: {} for i in VISION_V1_split_test_dict[data_split]}
    elif cfg.DATASET.name in ["DS_Spectrum_DS", "DS_Spectrum_DS_ND"]:
        result_dict = {i: {} for i in DS_Spectrum_DS_split_test_dict[data_split]}
    elif cfg.DATASET.name in ["openset_test_dataset", "openset_test_dataset_ND"]:
        result_dict = {i: {} for i in cfg.DATASET.open_set_test_object}
    else:
        raise NotImplementedError

    test_num = 0
    for i, data in enumerate(val_loader):

        s_input = data["support_image"]
        s_mask = data["support_mask"]
        input = data["query_image"]
        target = data["query_mask"]
        query_original_shape = data["query_original_shape"]
        query_input_shape = data["query_input_shape"]
        query_object_category_filename = data["query_object_category_filename"]
        support_img_path = data["support_img_path"]

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        start_time = time.time()
        if current_method in ["SOFS"]:
            output = model(s_x=s_input, s_y=s_mask, x=input)
        else:
            raise NotImplementedError
        model_time.update(time.time() - start_time)

        if current_method in ["SOFS"]:
            output_absolute_val = output.max(1)[1]
            # heatmap
            output_heatmap = output[:, 1, ...]

        for tmp_ocf, tmp_os, tmp_is, tmp_oav, tmp_ohm, tmp_mask, sip in zip(query_object_category_filename,
                                                                            query_original_shape,
                                                                            query_input_shape,
                                                                            output_absolute_val,
                                                                            output_heatmap,
                                                                            target,
                                                                            support_img_path
                                                                            ):
            query_object, query_category, query_filename = tmp_ocf.split("^")
            if int(query_category) not in result_dict[query_object].keys():
                result_dict[query_object][int(query_category)] = {
                    "intersection": [],
                    "union": [],
                    "new_target": []
                }

            if not train_validate:
                input_h, input_w = tmp_is.numpy()
                input_h, input_w = int(input_h), int(input_w)

                original_input_h, original_input_w = tmp_os.numpy()
                original_input_h, original_input_w = int(original_input_h), int(original_input_w)
                original_mask = tmp_mask.squeeze(0)[:original_input_h, :original_input_w]

                oav = upsample_output_result(
                    tmp_img=tmp_oav,
                    input_h=input_h,
                    input_w=input_w,
                    original_input_h=original_input_h,
                    original_input_w=original_input_w,
                    quantization=True
                )

                ohm = upsample_output_result(
                    tmp_img=tmp_ohm,
                    input_h=input_h,
                    input_w=input_w,
                    original_input_h=original_input_h,
                    original_input_w=original_input_w,
                    quantization=False
                )
            else:
                original_mask = tmp_mask.squeeze(0)
                oav = tmp_oav
                ohm = tmp_ohm

            intersection, union, target = intersectionAndUnionGPU(oav, original_mask, 2, 255)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            result_dict[query_object][int(query_category)]["intersection"].append(intersection)
            result_dict[query_object][int(query_category)]["union"].append(union)
            result_dict[query_object][int(query_category)]["new_target"].append(target)

            if np.random.rand(1)[0] < cfg.TEST.VISUALIZE.sample_prob:
                if cfg.TEST.VISUALIZE.save_figure:
                    test_num += 1
                    fig_save_path = os.path.join(cfg.OUTPUT_DIR, "figure_save")
                    os.makedirs(fig_save_path, exist_ok=True)

                    fig_save_path = os.path.join(fig_save_path, "_".join(["split",
                                                                          str(cfg.DATASET.split), "epoch_",
                                                                          str(epoch)]))
                    os.makedirs(fig_save_path, exist_ok=True)

                    produce_qualitative_result(
                        original_mask=original_mask,
                        oav=oav,
                        ohm=ohm,
                        source_path=cfg.TRAIN.dataset_path,
                        query_object=query_object,
                        query_filename=query_filename,
                        fig_save_path=fig_save_path,
                        test_num=test_num,
                        support_img_path=sip
                    )
    val_time = time.time() - val_start

    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND"]:
        result_dict_total = {i: {} for i in VISION_V1_split_test_dict[data_split]}
    elif cfg.DATASET.name in ["DS_Spectrum_DS", "DS_Spectrum_DS_ND"]:
        result_dict_total = {i: {} for i in DS_Spectrum_DS_split_test_dict[data_split]}
    elif cfg.DATASET.name in ["openset_test_dataset", "openset_test_dataset_ND"]:
        result_dict_total = {i: {} for i in cfg.DATASET.open_set_test_object}
    else:
        raise NotImplementedError

    result_dict_gather = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(result_dict_gather, result_dict)

    # if is_master_proc():
    for each_dict in result_dict_gather:
        for each_object, category_attribute_val in each_dict.items():
            for category_, attribute_val in category_attribute_val.items():
                if category_ not in result_dict_total[each_object].keys():
                    result_dict_total[each_object][category_] = {
                        "intersection": [],
                        "union": [],
                        "new_target": []
                    }

                for attribute_, val_ in attribute_val.items():
                    result_dict_total[each_object][category_][attribute_].extend(val_)

    class_iou_class = {}
    class_miou = 0
    FB_IOU_intersection = np.array([0., 0.])
    FB_IOU_union = np.array([0., 0.])

    class_miou, class_iou_class, FB_IOU = acquire_final_mIOU_FBIOU(
        result_dict=result_dict_total,
        class_iou_class=class_iou_class,
        class_miou=class_miou,
        FB_IOU_intersection=FB_IOU_intersection,
        FB_IOU_union=FB_IOU_union
    )

    LOGGER.info("current epoch is {}".format(epoch))
    LOGGER.info('meanIoU---Val result: mIoU_final {:.4f}.'.format(class_miou))

    LOGGER.info('<<<<<<< Every Class Results <<<<<<<')
    for i in class_iou_class.keys():
        LOGGER.info('{} Foreground Result: iou_f {:.4f}.'.format(i, class_iou_class[i]["foreground_iou"]))
        LOGGER.info('{} Background Result: iou_b {:.4f}.'.format(i, class_iou_class[i]["background_iou"]))
    for i in range(2):
        LOGGER.info('Class_{} Result: FB_IOU {:.4f}.'.format(i, FB_IOU[i]))

    LOGGER.info('FBIoU---Val result: FBIoU {:.4f}.'.format(np.mean(FB_IOU)))
    LOGGER.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    # todo
    # print('total time: {:.4f}, avg inference time: {:.4f}'.format(
    #     val_time,
    #     model_time.avg,
    #     test_num))
    # return class_miou


@torch.no_grad()
def epoch_validate_non_resize_ss(val_loader, model, epoch, cfg, rand_seed, mode="test"):
    LOGGER.info("Start validate!")
    model_time = AverageMeter()

    data_split = "split_" + str(cfg.DATASET.split)

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

    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND"]:
        if mode == "test":
            split_test_dict = VISION_V1_split_test_dict[data_split]
        else:
            split_test_dict = VISION_V1_split_train_val_dict[data_split]
    elif cfg.DATASET.name in ["DS_Spectrum_DS", "DS_Spectrum_DS_ND"]:
        split_test_dict = DS_Spectrum_DS_split_test_dict[data_split]

    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND", "DS_Spectrum_DS", "DS_Spectrum_DS_ND"]:
        result_dict = {i: {} for i in split_test_dict}
    elif cfg.DATASET.name in ["openset_test_dataset", "openset_test_dataset_ND"]:
        result_dict = {i: {} for i in cfg.DATASET.open_set_test_object}
    else:
        raise NotImplementedError

    test_num = 0

    for i, data in enumerate(val_loader):
        s_input = data["support_image"]
        s_mask = data["support_mask"]
        input = data["query_image"]
        target = data["query_mask"]
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
        target = target.cuda(non_blocking=True)

        if input.dim() == 4:
            start_time = time.time()
            if current_method in ["SOFS"]:
                output = model(s_x=s_input, s_y=s_mask, x=input)
            else:
                raise NotImplementedError

            model_time.update(time.time() - start_time)
            if current_method in ["SOFS"]:
                output_absolute_val = output.max(1)[1][0]
                # heatmap
                output_heatmap = output[:, 1, ...][0]

            original_output, original_heatmap = network_output2original_result(
                query_input_shape=query_input_shape,
                query_original_shape=query_original_shape,
                output_absolute_val=output_absolute_val,
                output_heatmap=output_heatmap
            )
        else:
            start_time = time.time()
            original_output = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_heatmap = torch.zeros(
                (grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_position = torch.zeros(
                (grid_size, int(query_original_shape[0]), int(query_original_shape[1])))

            multiple_ND = grid_size // cfg.TEST_SETUPS.ND_batch_size
            multiple_mod = grid_size % cfg.TEST_SETUPS.ND_batch_size
            multiple_bs = cfg.TEST_SETUPS.ND_batch_size
            iter_num = multiple_ND if multiple_mod == 0 else multiple_ND + 1

            start_time = time.time()
            if current_method in ["SOFS"]:
                output_total = []
                for pointer_idx in range(iter_num):
                    init_val = pointer_idx * cfg.TEST_SETUPS.ND_batch_size
                    _, gs, c, h, w = input.shape
                    if multiple_ND == 0:
                        multiple_bs = gs

                    if pointer_idx == multiple_ND:
                        multiple_bs = multiple_mod

                    support_input_m = s_input.repeat(multiple_bs, 1, 1, 1, 1)
                    support_mask_m = s_mask.repeat(multiple_bs, 1, 1, 1, 1)
                    query_input_m = input[:, init_val: init_val + multiple_bs, ...].reshape(1 * multiple_bs, c, h, w)

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
                else:
                    raise NotImplementedError

                if current_method in ["SOFS"]:
                    output_absolute_val = output.max(1)[1][0]
                    # heatmap
                    output_heatmap = output[:, 1, ...][0]

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

        query_object, query_category, query_filename = query_object_category_filename.split("^")
        if int(query_category) not in result_dict[query_object].keys():
            result_dict[query_object][int(query_category)] = {
                "intersection": [],
                "union": [],
                "new_target": []
            }

        original_mask = target.squeeze(0).squeeze(0)[:int(query_original_shape[0]), :int(query_original_shape[1])]

        intersection, union, target = intersectionAndUnionGPU(original_output.cuda(), original_mask, 2, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        result_dict[query_object][int(query_category)]["intersection"].append(intersection)
        result_dict[query_object][int(query_category)]["union"].append(union)
        result_dict[query_object][int(query_category)]["new_target"].append(target)

        if np.random.rand(1)[0] < cfg.TEST.VISUALIZE.sample_prob:
            if cfg.TEST.VISUALIZE.save_figure:
                test_num += 1
                fig_save_path = os.path.join(cfg.OUTPUT_DIR, "figure_save")
                os.makedirs(fig_save_path, exist_ok=True)

                fig_save_path = os.path.join(fig_save_path, "_".join(["split",
                                                                      str(cfg.DATASET.split), "epoch_",
                                                                      str(epoch)]))
                os.makedirs(fig_save_path, exist_ok=True)

                try:
                    produce_qualitative_result(
                        original_mask=original_mask,
                        oav=original_output,
                        ohm=original_heatmap,
                        source_path=cfg.TRAIN.dataset_path,
                        query_object=query_object,
                        query_filename=query_filename,
                        fig_save_path=fig_save_path,
                        test_num=test_num,
                        support_img_path=support_img_path
                    )
                except:
                    pass

    val_time = time.time() - val_start

    if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND", "DS_Spectrum_DS", "DS_Spectrum_DS_ND"]:
        result_dict_total = {i: {} for i in split_test_dict}
    elif cfg.DATASET.name in ["openset_test_dataset", "openset_test_dataset_ND"]:
        result_dict_total = {i: {} for i in cfg.DATASET.open_set_test_object}
    else:
        raise NotImplementedError

    result_dict_gather = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(result_dict_gather, result_dict)

    if is_master_proc():
        total_num = 0
        for each_dict in result_dict_gather:
            for each_object, category_attribute_val in each_dict.items():
                for category_, attribute_val in category_attribute_val.items():
                    if category_ not in result_dict_total[each_object].keys():
                        result_dict_total[each_object][category_] = {
                            "intersection": [],
                            "union": [],
                            "new_target": []
                        }

                    for idx, (attribute_, val_) in enumerate(attribute_val.items()):
                        if idx == 0:
                            total_num += len(val_)
                        result_dict_total[each_object][category_][attribute_].extend(val_)

        LOGGER.info("total test samples are {}".format(total_num))

        class_iou_class = {}
        class_miou = 0
        FB_IOU_intersection = np.array([0., 0.])
        FB_IOU_union = np.array([0., 0.])

        class_miou, class_iou_class, FB_IOU = acquire_final_mIOU_FBIOU(
            result_dict=result_dict_total,
            class_iou_class=class_iou_class,
            class_miou=class_miou,
            FB_IOU_intersection=FB_IOU_intersection,
            FB_IOU_union=FB_IOU_union
        )

        # if is_master_proc():
        LOGGER.info("current epoch is {}".format(epoch))
        LOGGER.info('meanIoU---Val result: mIoU_final {:.4f}.'.format(class_miou))

        LOGGER.info('<<<<<<< Every Class Results <<<<<<<')
        for i in class_iou_class.keys():
            LOGGER.info('{} Foreground Result: iou_f {:.4f}.'.format(i, class_iou_class[i]["foreground_iou"]))
            LOGGER.info('{} Background Result: iou_b {:.4f}.'.format(i, class_iou_class[i]["background_iou"]))
        for i in range(2):
            LOGGER.info('Class_{} Result: FB_IOU {:.4f}.'.format(i, FB_IOU[i]))

        LOGGER.info('FBIoU---Val result: FBIoU {:.4f}.'.format(np.mean(FB_IOU)))
        LOGGER.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')