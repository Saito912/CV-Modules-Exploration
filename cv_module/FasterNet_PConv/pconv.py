# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn

from torch import Tensor

from ..SeaFormer.seaformer import Sea_Attention_Block


class Partial_Conv(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(
            x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class Former_Partial_Conv(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_vit = dim // n_div
        self.dim_conv3 = dim - self.dim_vit
        self.partial_vit = Sea_Attention_Block(
            self.dim_vit, self.dim_vit//2, 4)
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_vit, :, :] = self.partial_vit(
            x[:, :self.dim_vit, :, :])
        x[:, self.dim_vit:, :, :] = self.partial_conv3(
            x[:, self.dim_vit:, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_vit, self.dim_conv3], dim=1)
        x1 = self.partial_vit(x1)
        x2 = self.partial_conv3(x2)
        x = torch.cat((x1, x2), 1)

        return x


if __name__ == "__main__":
    model = Former_Partial_Conv(256)
    inp = torch.randn(1, 256, 40, 40)
    oup = model(inp)
    # model_test(model, inp)
