import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, in_channels=(256, 512, 1024)):
        """
        Args:
            level(int): 选择特征图输出的尺寸, 0为20, 1为40, 2为80
            in_channels(Tuple[int]): 特征图的输入通道数
        """
        super(ASFF, self).__init__()
        self.level = level
        # 这一行本来不该存在的，一开始in_channels的顺序是从大到小，但是为了与forward输入的顺序对应，因此逆转in_channels
        in_channels = in_channels[::-1]
        self.dim = [int(in_channels[0] * multiplier),
                    int(in_channels[1] * multiplier),
                    int(in_channels[2] * multiplier)]

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(
                int(in_channels[1] * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(
                int(in_channels[2] * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                in_channels[0] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(in_channels[0] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(in_channels[2] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(
                in_channels[1] * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(in_channels[0] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(in_channels[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                in_channels[2] * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x: Tuple[torch.Tensor]):  # l,m,s
        """

        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
            level_1_resized * levels_weight[:, 1:2, :, :] + \
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
# ------------------------------------asff -----end--------------------------------


if __name__ == "__main__":
    from cv_module.model_test import model_test
    input_data = ((torch.randn(1, 256, 80, 80),
                   torch.randn(1, 512, 40, 40),
                   torch.randn(1, 1024, 20, 20)),)
    out = model_test(ASFF(1, in_channels=(256, 512, 1024)),
                     input_data=input_data)

    print(out.shape)
