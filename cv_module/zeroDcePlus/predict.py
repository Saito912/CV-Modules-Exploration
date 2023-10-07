import cv2
import numpy as np
import torch

from .model import enhance_net_nopool
import os
from torch import nn


class ZeroDceP(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        current_path = os.path.dirname(__file__)
        ckpt_path = os.path.join(current_path, 'zero_dce_plus.pth')
        print('load ckpt from ' + ckpt_path)
        self.model = enhance_net_nopool(8).to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, low_img):
        with torch.no_grad():
            self.model.eval()
            enhanced_image, _ = self.model(low_img)
            return enhanced_image
