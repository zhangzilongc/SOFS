import torch
import torch.nn.functional as F
from einops import rearrange


def conv_down_sample_vit(mask, patch_size=14):
    conv_param = torch.ones(patch_size, patch_size).cuda()
    down_sample_mask_vit = F.conv2d(
        mask,
        conv_param.unsqueeze(0).unsqueeze(0),
        stride=patch_size
    )
    down_sample_mask_vit = down_sample_mask_vit / (patch_size * patch_size)
    return down_sample_mask_vit


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask, patch_size=14, conv_vit_down_sampling=False):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    if conv_vit_down_sampling:
        mask = conv_down_sample_vit(mask, patch_size=patch_size)
    else:
        mask = F.interpolate(
            (mask == 1).float(), q.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0]
    similarity = similarity.view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_normal_similarity(tmp_q, tmp_s, mask, shot, patch_size=14, conv_vit_down_sampling=False):
    tmp_s = rearrange(tmp_s, "(b n) c h w -> b n c h w", n=shot)
    bs, shot, d, h, w = tmp_s.shape
    if conv_vit_down_sampling:
        tmp_mask = conv_down_sample_vit(mask, patch_size=patch_size)
    else:
        tmp_mask = F.interpolate(mask,
                                 size=(h, w),
                                 mode="bilinear",
                                 align_corners=False)
    tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b n 1 h w", n=shot)

    # b h*w c
    tmp_q = tmp_q.reshape(bs, d, -1).permute(0, 2, 1)

    tmp_s = tmp_s.reshape(bs, shot, d, -1).permute(0, 2, 1, 3).reshape(bs, d, -1).permute(
        0, 2, 1)
    tmp_mask = tmp_mask.reshape(bs, shot, 1, -1).permute(0, 2, 1, 3).reshape(bs, 1, -1)

    l2_normalize_s = F.normalize(tmp_s, dim=2)
    l2_normalize_q = F.normalize(tmp_q, dim=2)

    # b hw (n*hw)
    similarity = torch.bmm(l2_normalize_q, l2_normalize_s.permute(0, 2, 1))

    # for abnormal segmentation
    normal_similarity = similarity * (1 - tmp_mask)
    normal_cos_dis = 1 - normal_similarity.max(2)[0]

    min_max_abnormal_dis = normal_cos_dis.view(bs, 1, h, w)
    return min_max_abnormal_dis
