import os
import torch

import torch.nn as nn
from torch.optim import lr_scheduler

from lib.misc.optimizer import AdamW
from lib.misc.scheduler import WarmUpLR
from lib.misc import util

class CenterNet(object):
    def name(self):
        return 'CenterNet'

    @staticmethod
    def modify_commandline_options(parser, is_test):

        parser.add_argument('')
        return parser

    @staticmethod
    def get_common_args(cfg):
        common_args = dict()

        return common_args

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = cfg.logger
        self.gpu_ids = cfg.gpu_ids
        self.start_epoch = 0
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.device = torch.device(cfg.device)
        self.ckpt_dir = cfg.ckpt_dir
        self.expr_dir = os.path.join(cfg.ckpt_dir, cfg.expr_name)
        self.loss_names = []
        self.acc_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.training = not cfg.is_test
        self.optimizer = {}
        self.lr_scheduler = {}
        self.loss_meter = util.AverageMeter()

    def to_device(self, v):
        try:
            v = v.to(self.device)
        except AttributeError:
            v = v
        return v
