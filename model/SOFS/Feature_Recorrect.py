import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from model.SOFS.Transformer import Transformer_Nonlearnable_Fusion
from model.SOFS.utils import conv_down_sample_vit


class Feature_Recorrect_Module(nn.Module):

    def __init__(
            self,
            shot,
            fea_dim,
            reduce_dim,
            transformer_embed_dim,
            prior_layer_pointer,
            transformer_num_stages,
            transformer_nums_heads,
            cfg
    ):
        super(Feature_Recorrect_Module, self).__init__()
        self.shot = shot
        self.prior_layer_pointer = prior_layer_pointer
        self.patch_size = cfg.TRAIN.SOFS.vit_patch_size

        self.cfg = cfg

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_supp_semantic = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.supp_merge_semantic = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, transformer_embed_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        if cfg.TRAIN.SOFS.normal_sim_aug:
            num_similarity_channels = 2 * len(prior_layer_pointer)
        else:
            num_similarity_channels = len(prior_layer_pointer)

        self.query_merge_semantic = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + num_similarity_channels, transformer_embed_dim, kernel_size=1, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
        )

        self.query_semantic_transformer = Transformer_Nonlearnable_Fusion(
            shot=shot,
            temperature=cfg.TRAIN.SOFS.target_semantic_temperature,
            num_stages=transformer_num_stages,
            match_dims=transformer_embed_dim,
            match_nums_heads=transformer_nums_heads,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.,
            meta_cls=cfg.TRAIN.SOFS.meta_cls
        )

        if not cfg.TRAIN.SOFS.meta_cls:
            self.cls_semantic = nn.Sequential(
                nn.Conv2d(transformer_embed_dim, transformer_embed_dim * 4, kernel_size=1, stride=1, padding=0),
                nn.SyncBatchNorm(transformer_embed_dim * 4),
                nn.Conv2d(transformer_embed_dim * 4, transformer_embed_dim * 4, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(transformer_embed_dim * 4),
                nn.Conv2d(transformer_embed_dim * 4, 2, kernel_size=1, stride=1, padding=0)
            )

    def forward(self,
                query_features_list,
                support_features_list,
                supp_feat_bin_list,
                semantic_similarity,
                normal_similarity,
                mask,
                conv_vit_down_sampling=False
                ):
        """
        """
        tmp_query_feat = torch.cat(query_features_list, 1)
        tmp_supp_feat = torch.cat(support_features_list, 1)
        supp_feat_bin = torch.concat(supp_feat_bin_list, dim=1)

        tmp_query_feat = self.down_query(tmp_query_feat)
        tmp_supp_feat = self.down_supp(tmp_supp_feat)
        tmp_supp_feat_bin = self.down_supp_semantic(supp_feat_bin)

        tmp_supp_feat_merge = self.supp_merge_semantic(torch.cat([tmp_supp_feat, tmp_supp_feat_bin], dim=1))

        tmp_supp_feat_bin = rearrange(tmp_supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        tmp_supp_feat_bin = torch.mean(tmp_supp_feat_bin, dim=1)  # 对k-shot的取一个平均向量，b c h w

        if self.cfg.TRAIN.SOFS.normal_sim_aug:
            query_semantic_similarity = torch.concat([semantic_similarity, normal_similarity], dim=1)
        else:
            query_semantic_similarity = semantic_similarity

        # follow HDMNet 10*
        tmp_query_feat_semantic = self.query_merge_semantic(torch.cat([tmp_query_feat,
                                                                       tmp_supp_feat_bin,
                                                                       10 * query_semantic_similarity], dim=1))

        _, _, supp_h, supp_w = tmp_supp_feat_merge.shape
        if conv_vit_down_sampling:
            down_sample_mask = conv_down_sample_vit(mask, patch_size=self.patch_size)
        else:
            down_sample_mask = F.interpolate(mask,
                                             size=(supp_h, supp_w),
                                             mode="bilinear",
                                             align_corners=False)
        tmp_down_sample_mask = rearrange(down_sample_mask, "(b n) 1 h w -> b n 1 h w", n=self.shot)

        # b, d, h, w and b, d
        if self.cfg.TRAIN.SOFS.meta_cls:
            final_out_semantic, s_x_prototype = self.query_semantic_transformer(tmp_query_feat_semantic,
                                                                                tmp_supp_feat_merge,
                                                                                tmp_down_sample_mask)
        else:
            final_out_semantic = self.query_semantic_transformer(tmp_query_feat_semantic,
                                                                 tmp_supp_feat_merge,
                                                                 tmp_down_sample_mask)

        bs_q, q_d, q_h, q_w = final_out_semantic.shape

        if self.cfg.TRAIN.SOFS.meta_cls:
            final_out_semantic = (s_x_prototype.unsqueeze(1) @ final_out_semantic.view(bs_q, q_d, q_h * q_w)).view(bs_q,
                                                                                                                   1,
                                                                                                                   q_h,
                                                                                                                   q_w)
        else:
            final_out_semantic = self.cls_semantic(final_out_semantic)

        if self.cfg.TRAIN.SOFS.meta_cls:
            final_out = final_out_semantic[:, 0, ...]
        else:
            final_out = final_out_semantic
        return final_out
