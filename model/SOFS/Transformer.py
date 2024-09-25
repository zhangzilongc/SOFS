import math
from einops import rearrange
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from model.SOFS.MaskMultiheadAttention import MaskMultiHeadAttention
from model.SOFS.utils import Weighted_GAP
import torch.nn.functional as F


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MaskMultiHeadAttention(
            in_features=embed_dims, head_num=num_heads, bias=False, activation=None
        )

    def forward(self, x, hw_shape, source=None, identity=None, mask=None, cross=False):
        x_q = x
        if source is None:
            x_kv = x
        else:
            x_kv = source
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x_kv, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)

        if identity is None:
            identity = x_q

        out = self.attn(q=x_q, k=x_kv, v=x_kv, mask=mask, cross=cross)
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio
        )

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape, source=None, mask=None, cross=False):
        if source is None:
            x = self.attn(self.norm1(x), hw_shape, identity=x)
        else:
            x = self.attn(self.norm1(x), hw_shape, source=self.norm1(source), identity=x, mask=mask, cross=cross)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class MixVisionTransformer(BaseModule):
    def __init__(self,
                 shot=1,
                 temperature=0.1,
                 num_stages=2,
                 match_dims=64,
                 match_nums_heads=2,
                 mlp_ratio=4,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 qkv_bias=False,
                 meta_cls=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)

        self.shot = shot
        self.temperature = temperature
        self.num_stages = num_stages
        self.match_dims = match_dims
        self.match_nums_heads = match_nums_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.meta_cls = meta_cls

        self.sa_layers = ModuleList()

        for i in range(self.num_stages):
            self.sa_layers.append(
                TransformerEncoderLayer(
                    embed_dims=match_dims,
                    num_heads=match_nums_heads,
                    feedforward_channels=mlp_ratio * match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
            )

        if meta_cls:
            self.support_prototype = nn.Conv2d(self.match_dims, self.match_dims, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()

    def forward(self, q_x, s_x, mask):
        """

        :param q_x: n c h w
        :param s_x: n k c h w
        :param mask: n k 1 h w
        :return:
        """
        bs, d, h, w = q_x.shape
        bs_s, d, h, w = s_x.shape
        hw_shape = (h, w)

        tmp_mask = mask.reshape(bs_s, 1, h, w)
        mask = mask.reshape(bs, self.shot, 1, -1).permute(0, 2, 1, 3).reshape(bs, 1, -1)

        # b hw d
        q_x = q_x.reshape(bs, d, -1).permute(0, 2, 1)

        s_x = s_x.reshape(bs_s, d, -1).permute(0, 2, 1)
        for i in range(self.num_stages):
            q_x = self.sa_layers[i](q_x, hw_shape=hw_shape)
            s_x = self.sa_layers[i](s_x, hw_shape=hw_shape)

        s_x = s_x.reshape(bs_s, h * w, d).permute(0, 2, 1).reshape(bs_s, d, h, w)

        if self.meta_cls:
            # encode prototype
            s_x_prototype = self.support_prototype(s_x)
            s_x_prototype = Weighted_GAP(s_x_prototype, tmp_mask)  # bs_s, d, 1, 1
            # s_x_prototype = Weighted_GAP(s_x, tmp_mask)  # bs_s, d, 1, 1
            s_x_prototype = torch.mean(s_x_prototype.reshape(bs, self.shot, d, 1, 1), dim=1).squeeze(-1).squeeze(-1)

        s_x = rearrange(s_x, "(b n) c h w -> b n c h w", n=self.shot)

        # b (shot*hw) d
        # q_x = q_x.reshape(bs, d, -1).permute(0, 2, 1)
        s_x = s_x.reshape(bs, self.shot, d, -1).permute(0, 2, 1, 3).reshape(bs, d, -1).permute(0, 2, 1)

        q_init_x = q_x.clone()
        normalized_query, normalized_key = F.normalize(q_init_x, dim=2), F.normalize(s_x, dim=2)
        # b hw (n*hw)
        similarity_ = torch.einsum("bmc,bnc->bmn", normalized_query, normalized_key)

        # Nonlearnable_Fusion
        semantic_similarity = similarity_ * mask
        attention = torch.softmax(semantic_similarity / self.temperature, dim=1)
        q_under_s = torch.bmm(attention, s_x * mask.permute(0, 2, 1))
        q_init_x = q_init_x + q_under_s
        q_init_x = q_init_x.permute(0, 2, 1).reshape(bs, d, h, w)

        if self.meta_cls:
            return q_init_x, s_x_prototype
        else:
            return q_init_x


# modified based on HDMNet
class Transformer_Nonlearnable_Fusion(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mix_transformer = MixVisionTransformer(**kwargs)

    def forward(self, features, supp_features, mask):
        outs = self.mix_transformer(features, supp_features, mask)
        return outs
