# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from projects.centerTrack.fastreid.config import CfgNode as CN


def add_moco_config(cfg):
    _C = cfg

    _C.MODEL.HEADS.CAM_CLASSES = 44
    _C.MODEL.HEADS.CAM_FEAT = 256
    _C.MODEL.HEADS.DOMAIN_CLASSES = 15

    # Memory bank
    _C.MODEL.MEMORY = CN()
    _C.MODEL.MEMORY.QUEUE_SIZE = 8192
    _C.MODEL.MEMORY.FEAT_DIM = 512
    _C.MODEL.MEMORY.MOMENTUM = 0.999

    _C.MODEL_TEACHER = CN()
    _C.MODEL_TEACHER.META_ARCHITECTURE = 'Baseline'

    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.BACKBONE = CN()

    _C.MODEL_TEACHER.BACKBONE.NAME = "build_resnet_backbone"
    _C.MODEL_TEACHER.BACKBONE.DEPTH = "50x"
    _C.MODEL_TEACHER.BACKBONE.LAST_STRIDE = 1
    # Input feature dimension
    _C.MODEL_TEACHER.BACKBONE.FEAT_DIM = 2048
    # If use IBN block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_IBN = True
    # If use SE block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_SE = False
    # If use Non-local block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_NL = False

    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.HEADS = CN()
    _C.MODEL_TEACHER.HEADS.NAME = "EmbeddingHead"

    # Pooling layer type
    _C.MODEL_TEACHER.HEADS.POOL_LAYER = "gempool"
    _C.MODEL_TEACHER.HEADS.NECK_FEAT = "after"
    _C.MODEL_TEACHER.HEADS.CLS_LAYER = "CircleSoftmax"

    # Pretrained teacher and student model weights
    _C.MODEL.TEACHER_WEIGHTS = ""
    _C.MODEL.STUDENT_WEIGHTS = ""


def update_model_teacher_config(cfg):
    cfg = cfg.clone()

    frozen = cfg.is_frozen()

    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = cfg.MODEL_TEACHER.META_ARCHITECTURE
    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.BACKBONE.NAME = cfg.MODEL_TEACHER.BACKBONE.NAME
    cfg.MODEL.BACKBONE.DEPTH = cfg.MODEL_TEACHER.BACKBONE.DEPTH
    cfg.MODEL.BACKBONE.LAST_STRIDE = cfg.MODEL_TEACHER.BACKBONE.LAST_STRIDE
    # Input feature dimension
    cfg.MODEL.BACKBONE.FEAT_DIM = cfg.MODEL_TEACHER.BACKBONE.FEAT_DIM
    # If use IBN block in backbone
    cfg.MODEL.BACKBONE.WITH_IBN = cfg.MODEL_TEACHER.BACKBONE.WITH_IBN
    # If use SE block in backbone
    cfg.MODEL.BACKBONE.WITH_SE = cfg.MODEL_TEACHER.BACKBONE.WITH_SE
    # If use Non-local block in backbone
    cfg.MODEL.BACKBONE.WITH_NL = cfg.MODEL_TEACHER.BACKBONE.WITH_NL
    cfg.MODEL.BACKBONE.PRETRAIN = False
    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.HEADS.NAME = cfg.MODEL_TEACHER.HEADS.NAME

    # Pooling layer type
    cfg.MODEL.HEADS.POOL_LAYER = cfg.MODEL_TEACHER.HEADS.POOL_LAYER

    if frozen: cfg.freeze()

    return cfg
