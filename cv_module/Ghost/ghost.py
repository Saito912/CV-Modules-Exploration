from torch import nn
import torch
import math


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def DWConv(in_channels, out_channels, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(in_channels, out_channels, k, s, g=math.gcd(in_channels, out_channels), act=act)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    # ch_in, ch_out, kernel, stride, groups
    def __init__(self, in_channels, out_channels, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = out_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Ghost(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, in_channels, out_channels, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(Ghost, self).__init__()
        c_ = out_channels // 2
        self.conv = nn.Sequential(GhostConv(in_channels, c_, 1, 1),  # pw
                                  # dw
                                  DWConv(
                                      c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
                                  GhostConv(c_, out_channels, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(in_channels, in_channels, k, s, act=False),
                                      Conv(in_channels, out_channels, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


if __name__ == "__main__":
    from cv_module.model_test import model_test
    model = GhostConv(256, 256)
    model_test(model, (torch.randn(1, 256, 20, 20),))
