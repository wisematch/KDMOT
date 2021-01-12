from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .fastreid.modeling.meta_arch.build import build_model as build_fast_reid
from .fastreid.utils.checkpoint import Checkpointer as FastReID_Checkpointer
from .fastreid.config.config import get_cfg as get_fastreid_cfg

def create_teacher_model():
    cfg = get_fastreid_cfg()
    embedding_model = build_fast_reid(cfg)
    return embedding_model

def load_teacher_model(model_t, model_t_path):
    FastReID_Checkpointer(model_t).load(model_t_path)

    for param in model_t.parameters():
        param.requires_grad = False

    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    model_t.apply(set_bn_eval)
    model_t.eval()

    return model_t