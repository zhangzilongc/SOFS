import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.common import freeze_paras, ForwardHook, dice_ce_loss_sum
from utils import load_backbones
from einops import rearrange

from model.SOFS.Feature_Recorrect import Feature_Recorrect_Module
from model.SOFS.utils import Weighted_GAP, get_similarity, get_normal_similarity, conv_down_sample_vit


class SOFS(nn.Module):

    def __init__(self, cfg):
        super(SOFS, self).__init__()

        if cfg.DATASET.name in ["VISION_V1_ND", "DS_Spectrum_DS_ND", "opendomain_test_dataset_ND", 'ECCV_Contest_ND', "ECCV_Contest_Test_ND"]:
            shot = cfg.DATASET.shot * cfg.DATASET.s_in_shot
        else:
            shot = cfg.DATASET.shot

        prior_layer_pointer = cfg.TRAIN.SOFS.prior_layer_pointer

        backbone = load_backbones(cfg.TRAIN.backbone)

        if cfg.TRAIN.backbone_load_state_dict:
            # for vit
            state_dict = torch.load(cfg.TRAIN.backbone_checkpoint, map_location=torch.device("cpu"))
            backbone.load_state_dict(state_dict, strict=False)
        freeze_paras(backbone)

        self.outputs = {}
        for tmp_layer in prior_layer_pointer:
            forward_hook = ForwardHook(
                self.outputs, tmp_layer
            )
            if cfg.TRAIN.backbone in ['dinov2_vitb14', 'dinov2_vitl14']:
                network_layer = backbone.__dict__["_modules"]["blocks"][tmp_layer]
            elif cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                network_layer = backbone.__dict__["_modules"]["layer" + str(tmp_layer)]
            else:
                raise NotImplementedError

            if isinstance(network_layer, torch.nn.Sequential):
                network_layer[-1].register_forward_hook(forward_hook)
            else:
                network_layer.register_forward_hook(forward_hook)

        self.backbone = backbone
        self.shot = shot
        self.prior_layer_pointer = prior_layer_pointer
        self.target_semantic_temperature = cfg.TRAIN.SOFS.target_semantic_temperature
        self.ce_weight = cfg.TRAIN.LOSS.ce_weight
        self.dice_weight = cfg.TRAIN.LOSS.dice_weight

        if cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
            from utils.common import PatchMaker
            self.patch_maker = PatchMaker(3, stride=1)
            self.preprocessing_dim = [1024, 1024]
        elif cfg.TRAIN.backbone in ['dinov2_vitb14']:
            self.preprocessing_dim = [768] * len(prior_layer_pointer)
        elif cfg.TRAIN.backbone in ['dinov2_vitl14']:
            self.preprocessing_dim = [1024] * len(prior_layer_pointer)

        self.cfg = cfg

        reduce_dim = cfg.TRAIN.SOFS.reduce_dim
        fea_dim = sum(self.preprocessing_dim)
        transformer_embed_dim = cfg.TRAIN.SOFS.transformer_embed_dim
        transformer_num_stages = cfg.TRAIN.SOFS.transformer_num_stages
        transformer_nums_heads = cfg.TRAIN.SOFS.transformer_nums_heads

        self.feature_recorrect = Feature_Recorrect_Module(
            shot=shot,
            fea_dim=fea_dim,
            reduce_dim=reduce_dim,
            transformer_embed_dim=transformer_embed_dim,
            prior_layer_pointer=prior_layer_pointer,
            transformer_num_stages=transformer_num_stages,
            transformer_nums_heads=transformer_nums_heads,
            cfg=cfg
        )

    def encode_feature(self, x):
        self.outputs.clear()
        with torch.no_grad():
            _ = self.backbone(x)
        multi_scale_features = [self.outputs[key]
                                for key in self.prior_layer_pointer]
        return multi_scale_features

    @torch.no_grad()
    def feature_processing_vit(self, features, mask=None):
        B, L, C = features[0][:, 1:, :].shape
        h = w = int(math.sqrt(L))
        multi_scale_features = [each_feature[:, 1:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
                                for each_feature in features]
        if mask is not None:
            # due to the missing mask, we do not use the mask in this stage for HDM and PFE
            multi_scale_features_ = []
            for each_feature in multi_scale_features:
                tmp_mask = F.interpolate(mask,
                                         size=(each_feature.size(2),
                                               each_feature.size(3)),
                                         mode="bilinear",
                                         align_corners=False)
                multi_scale_features_.append(each_feature * tmp_mask)
            return multi_scale_features_
        else:
            return multi_scale_features

    def feature_processing_cnn(self, features):
        bs, _, h, w = features[0].shape
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]  # [((bs, h*w, c, 3, 3), [h, w]), ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # (bs, h, w, c, 3, 3)
            _features = _features.permute(0, -3, -2, -1, 1, 2)  # (bs, c, 3, 3, h, w)
            perm_base_shape = _features.shape  # (bs, c, 3, 3, h, w)
            _features = _features.reshape(-1, *_features.shape[-2:])  # (bs * c * 3 * 3, h, w)
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )  # (bs, c, 3, 3, h*2, w*2)
            _features = _features.permute(0, -2, -1, 1, 2, 3)  # (bs, h*2, w*2, c, 3, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])  # (bs, h*w, c, 3, 3)
            features[i] = _features
        # [[bs * h * w, c, 3, 3], [bs * (h1 * 2)=h * (w1 * 2)=w, c*2, 3, 3]]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]  # (bs * h * w, c, 3, 3)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = [F.adaptive_avg_pool1d(x.reshape(len(x), 1, -1), self.preprocessing_dim[idx]) for idx, x in
                    enumerate(features)]
        features = torch.concat(features, dim=1)
        features = features.reshape(len(features), 1, -1)  # bs * h * w, 1, 2 * self.preprocessing_dim
        features = F.adaptive_avg_pool1d(features, self.preprocessing_dim[-1])
        features = [features.reshape(-1, h, w, self.preprocessing_dim[-1]).permute(0, 3, 1, 2)] * len(
            self.prior_layer_pointer)
        return features

    def generate_query_label(self, x, s_x, s_y):
        x_size = x.size()
        bs_q, _, img_ori_h, img_ori_w = x_size
        patch_size = self.cfg.TRAIN.SOFS.vit_patch_size
        conv_vit_down_sampling = self.cfg.TRAIN.SOFS.conv_vit_down_sampling

        with torch.no_grad():
            query_multi_scale_features = self.encode_feature(x)
            query_features_list = []

            if self.cfg.TRAIN.backbone in ['dinov2_vitb14', "dinov2_vitl14"]:
                query_features = self.feature_processing_vit(query_multi_scale_features)
            elif self.cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                query_features = self.feature_processing_cnn(query_multi_scale_features)
            for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                exec("query_feat_{}=query_features[{}]".format(layer_pointer, idx))
                query_features_list.append(eval('query_feat_' + str(layer_pointer)))

            #   Support Feature
            mask = rearrange(s_y, "b n 1 h w -> (b n) 1 h w")
            mask = (mask == 1.).float()
            s_x = rearrange(s_x, "b n c h w -> (b n) c h w")
            support_multi_scale_features = self.encode_feature(s_x)
            support_features_list = []
            if self.cfg.TRAIN.backbone in ['dinov2_vitb14', "dinov2_vitl14"]:
                support_features = self.feature_processing_vit(support_multi_scale_features)
            elif self.cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                support_features = self.feature_processing_cnn(support_multi_scale_features)
            for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                exec("supp_feat_{}=support_features[{}]".format(layer_pointer, idx))
                support_features_list.append(eval('supp_feat_' + str(layer_pointer)))

            # weighted GAP for every layer
            supp_feat_bin_list = []
            for each_layer_supp_feat in support_features_list:
                if conv_vit_down_sampling:
                    tmp_mask = conv_down_sample_vit(mask, patch_size=patch_size)
                else:
                    tmp_mask = F.interpolate(
                        mask,
                        size=(each_layer_supp_feat.size(2),
                              each_layer_supp_feat.size(3)),
                        mode="bilinear",
                        align_corners=False
                    )
                supp_feat_bin = Weighted_GAP(
                    each_layer_supp_feat,
                    tmp_mask
                )
                supp_feat_bin = supp_feat_bin.repeat(1, 1, each_layer_supp_feat.shape[-2],
                                                     each_layer_supp_feat.shape[-1])
                supp_feat_bin_list.append(supp_feat_bin)

            # semantic similarity
            if self.shot == 1:
                similarity2 = []
                for layer_pointer in self.prior_layer_pointer:
                    similarity2.append(get_similarity(eval('query_feat_' + str(layer_pointer)),
                                                      eval('supp_feat_' + str(layer_pointer)),
                                                      mask,
                                                      patch_size=patch_size,
                                                      conv_vit_down_sampling=conv_vit_down_sampling))  # b c h w, (bn) c h w, (bn) 1 h w --> b 1 h w
                semantic_similarity = torch.concat(similarity2, dim=1)
            else:
                mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
                layer_similarity = []
                for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                    tmp_supp_feat = rearrange(eval('supp_feat_' + str(layer_pointer)), "(b n) c h w -> b n c h w",
                                              n=self.shot)
                    similarity2 = []
                    for i in range(self.shot):
                        similarity2.append(get_similarity(eval('query_feat_' + str(layer_pointer)),
                                                          tmp_supp_feat[:, i, ...],
                                                          mask=mask[:, i, ...],
                                                          patch_size=patch_size,
                                                          conv_vit_down_sampling=conv_vit_down_sampling))

                    similarity2 = torch.stack(similarity2, dim=1).mean(1)
                    layer_similarity.append(similarity2)
                mask = rearrange(mask, "b n c h w -> (b n) c h w")
                semantic_similarity = torch.concat(layer_similarity, dim=1)

            # normal similarity
            layer_out = []
            for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                tmp_s = eval('supp_feat_' + str(layer_pointer))
                tmp_q = eval('query_feat_' + str(layer_pointer))

                abnormal_dis = get_normal_similarity(tmp_q, tmp_s, mask, self.shot, patch_size=patch_size, conv_vit_down_sampling=conv_vit_down_sampling)
                layer_out.append(abnormal_dis)
            normal_similarity = torch.concat(layer_out, dim=1)
            each_normal_similarity = (normal_similarity.max(1)[0]).unsqueeze(1)

            mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
            mask_weight = mask.reshape(bs_q, -1).sum(1)
            mask_weight = (mask_weight > 0).float()
            mask = rearrange(mask, "b n c h w -> (b n) c h w")

        final_out = self.feature_recorrect(
            query_features_list=query_features_list,
            support_features_list=support_features_list,
            supp_feat_bin_list=supp_feat_bin_list,
            semantic_similarity=semantic_similarity,
            normal_similarity=normal_similarity,
            mask=mask,
            conv_vit_down_sampling=conv_vit_down_sampling
        )

        return final_out, mask_weight, each_normal_similarity

    def forward(self, x, s_x, s_y, y=None):
        x_size = x.size()
        bs_q, _, img_ori_h, img_ori_w = x_size
        patch_size = self.cfg.TRAIN.SOFS.vit_patch_size
        conv_vit_down_sampling = self.cfg.TRAIN.SOFS.conv_vit_down_sampling
        final_out, mask_weight, each_normal_similarity = self.generate_query_label(x, s_x, s_y)

        if self.training:
            _h, _w = final_out.shape[-2:]
            if conv_vit_down_sampling:
                y_m_squeeze = conv_down_sample_vit(y, patch_size=patch_size).squeeze(1)
            else:
                y_m_squeeze = F.interpolate(y, size=(_h, _w), mode='bilinear', align_corners=False).squeeze(1)

            y_m_squeeze = (y_m_squeeze > 0.1).float()
            if self.cfg.TRAIN.SOFS.meta_cls:
                final_out_prob = torch.sigmoid(final_out).contiguous()
            else:
                final_out_prob = torch.softmax(final_out, dim=1)[:, 1, ...].contiguous()

            main_loss = dice_ce_loss_sum(
                y_m_squeeze=y_m_squeeze,
                final_out=final_out_prob,
                dice_weight=self.dice_weight,
                ce_weight=self.ce_weight,
                smooth_r=self.cfg.TRAIN.SOFS.smooth_r
            )

            if self.cfg.TRAIN.SOFS.meta_cls:
                final_out = F.interpolate(final_out.unsqueeze(1), size=(img_ori_h, img_ori_w), mode='bilinear',
                                          align_corners=False).squeeze(1)
                final_out_prob = torch.sigmoid(final_out).contiguous()
            else:
                final_out = F.interpolate(final_out, size=(img_ori_h, img_ori_w), mode='bilinear', align_corners=False)
                final_out_prob = torch.softmax(final_out, dim=1)[:, 1, ...].contiguous()

            final_out = torch.cat([1 - final_out_prob.unsqueeze(1), final_out_prob.unsqueeze(1)], dim=1)
            return final_out.max(1)[1], main_loss
        else:
            mask_weight_ = mask_weight.unsqueeze(1).unsqueeze(1)
            normal_out = F.interpolate(each_normal_similarity, size=(img_ori_h, img_ori_w), mode='bilinear',
                                       align_corners=False).squeeze(1)
            # each_normal_similarity_
            if self.cfg.TRAIN.SOFS.meta_cls:
                final_out = F.interpolate(final_out.unsqueeze(1), size=(img_ori_h, img_ori_w), mode='bilinear',
                                          align_corners=False).squeeze(1)
                final_out_prob = torch.sigmoid(final_out).contiguous()
            else:
                final_out = F.interpolate(final_out, size=(img_ori_h, img_ori_w), mode='bilinear', align_corners=False)
                final_out_prob = torch.softmax(final_out, dim=1)[:, 1, ...].contiguous()

            final_out_prob = mask_weight_ * final_out_prob + (1 - mask_weight_) * normal_out

            final_out = torch.cat([1 - final_out_prob.unsqueeze(1), final_out_prob.unsqueeze(1)], dim=1)
            return final_out
