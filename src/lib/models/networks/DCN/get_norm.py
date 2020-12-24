from __future__ import absolute_import


import torch.nn as nn


def get_norm(cfg, out_channels, momentum=0.1):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    norm = cfg.MODEL.BN_TYPE
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels, momentum=momentum)