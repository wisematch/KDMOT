from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F


class KD_RegLoss(nn.Module):
    def __init__(self):
        super(KD_RegLoss, self).__init__()

    def forward(self, pred, target, mask):
        loss = _reg_loss(pred, target)

        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()

        regr = regr * mask
        gt_regr = gt_regr * mask

        regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss