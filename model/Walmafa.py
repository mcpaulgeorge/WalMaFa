
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
from visualizer import get_local
import numpy as np

from model import blocks
from mamba_ssm import Mamba

m = None


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  ##返回所有元素的方差
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x, ill_map, mask=None):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh w) h dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        ill = reshape(ill_map)
        v = v * ill
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac

        # 置为-1e9 因为softmax中0输出为1，我们想要输出为0就是负无穷大
        if mask is not None:
            mask = reshape(mask)
            mask = torch.matmul(mask, mask.transpose(-2, -1)) * self.fac
            res = res.masked_fill(mask == 0, -1e9)
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh w) h dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor, ill_map_height, ill_map_width, mask=None):
        assert len(x.size()) == 4
        x = self.row_att(x, ill_map_width, mask=mask)
        x = x.transpose(-2, -1)
        if mask is not None:
            x = self.col_att(x, ill_map_height.transpose(-2, -1), mask=mask.transpose(-2, -1), )
        else:
            x = self.col_att(x, ill_map_height.transpose(-2, -1), mask=mask)
        x = x.transpose(-2, -1)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.rep_conv1 = Conv2d_BN(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.rep_conv2 = Conv2d_BN(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x2 = self.dwconv(x)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity

    @torch.no_grad()
    def fuse(self):
        conv = self.rep_conv1.fuse()  ##Conv_BN
        conv1 = self.rep_conv2.fuse()  ##Conv_BN

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class WM(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=9,  # Block expansion factor
        )
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c))
       
        y = self.model1(x).permute(0, 2, 1) 
        output = y.reshape(b, c, h, w)
        return self.smooth(output)


class WMB(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(WMB, self).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.illu = Illumination_Estimator(dim, n_fea_in=dim+1, n_fea_out=dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.mb = WM(dim)

    def forward(self, input_):
        global m
        x = input_
        n, c, h, w = x.shape
        x = self.norm1(x)
        x = data_transform(x)
        input_dwt = self.DWT(x)
        # input_LL=A [B,C,H/2,W/2]   input_high0={V,H,D} [3B,C,H/2,W/2]
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        input_LL, input_image = self.illu(input_LL)
        input_high = self.mb(input_high)

        output = self.IWT(torch.cat((input_LL, input_high), dim=0))
        output = inverse_data_transform(output)

        x = x + output
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class Walmafa(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1, 2, 4],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 attention=True,
                 skip=False
                 ):

        super(Walmafa, self).__init__()

        self.estimator = Illumination_Estimator(dim)

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((4, 2, int(int(dim * 2 * 4))))),
                                        requires_grad=attention)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_mask = OverlapPatchEmbed(1, dim)

        self.sim = nn.Sequential(*[
            WMB(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_1 = Downsample(int(dim))  # From Level 0 to Level 1

        self.decoder_level1_0 = nn.Sequential(*[WMB(dim=int(int(dim * 2)), num_heads=heads[1],
                                                    ffn_expansion_factor=ffn_expansion_factor,
                                                    bias=bias, LayerNorm_type=LayerNorm_type)
                                                for i in range(num_blocks[0])])

        self.down_2 = Downsample(int(dim * 2))  # From Level 1 to Level 2
        self.decoder_level2_0 = nn.Sequential(*[
            WMB(dim=int(int(dim * 2 * 2)), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down_3 = Downsample(int(dim * 2 * 2))  # From Level 2 to Level 3
        self.decoder_level3_0 = nn.Sequential(*[
            WMB(dim=int(int(dim * 2 * 4)), num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent = blocks.FFAB(int(int(dim * 2 * 4)))

        self.up3_2 = Upsample(int(dim * 2 * 4))  # From Level 3 to Level 2
        self.decoder_level2_1 = nn.Sequential(*[
            WMB(dim=int(int(dim * 2 * 2)), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 * 2))  # From Level 2 to Level 1
        self.decoder_level1_1 = nn.Sequential(*[WMB(dim=int(int(dim * 2)), num_heads=heads[1],
                                                    ffn_expansion_factor=ffn_expansion_factor,
                                                    bias=bias, LayerNorm_type=LayerNorm_type)
                                                for i in range(num_blocks[0])])
        self.up2_0 = Upsample(int(dim * 2))  # From Level 1 to Level 0
        # skip connection wit weights
        self.coefficient_3_2 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 2))))), requires_grad=attention)
        self.coefficient_2_1 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2))))), requires_grad=attention)
        self.coefficient_1_0 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim))))), requires_grad=attention)

        # skip then conv 1x1
        self.skip_3_2 = nn.Conv2d(int(int(dim * 2 * 2)), int(int(dim * 2 * 2)), kernel_size=1, bias=bias)
        self.skip_2_1 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)
        self.skip_1_0 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)

        self.sim = nn.Sequential(*[
            WMB(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip

    def forward(self, inp_img):
        illu_fea, illu_map = self.estimator(inp_img)
        inp_img = inp_img * illu_map + inp_img
        inp_enc_encoder1 = self.patch_embed(inp_img)

        inp_enc_level1_0 = self.down_1(inp_enc_encoder1)
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)

        inp_enc_level2_0 = self.down_2(out_enc_level1_0)
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)

        inp_enc_level3_0 = self.down_3(out_enc_level2_0)
        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)

        out_enc_level3_0 = self.latent(out_enc_level3_0)

        out_enc_level3_0 = self.up3_2(out_enc_level3_0)
        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0 + self.coefficient_3_2[1,
                                                                                                :][None, :, None,
                                                                                                None] * out_enc_level3_0
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)  ### conv 1x1
        out_enc_level2_1 = self.decoder_level2_1(inp_enc_level2_1)

        out_enc_level2_1 = self.up2_1(out_enc_level2_1)

        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_enc_level1_0 + self.coefficient_2_1[1,
                                                                                                :][None, :, None,
                                                                                                None] * out_enc_level2_1

        inp_enc_level1_1 = self.skip_1_0(inp_enc_level1_1)  ### conv 1x1
        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)
        out_enc_level1_1 = self.up2_0(out_enc_level1_1)
        out_fusion_123 = self.sim(out_enc_level1_1)

        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123 + self.coefficient_1_0[1, :][None, :,
                                                                                 None, None] * out_enc_level1_1

        if self.skip:
            out = self.output(out) + inp_img
        else:
            out = self.output(out)

        return out
