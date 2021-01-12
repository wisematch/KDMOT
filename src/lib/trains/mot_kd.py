from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from progress.bar import Bar
from utils.utils import AverageMeter

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss


from models.utils import _sigmoid, _tranpose_and_gather_feat

from .base_trainer import BaseTrainer



class MotKDLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotKDLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

        self.crit_kd = torch.nn.MSELoss(reduction='mean', size_average=False) if opt.kd_loss == 'mse' else \
            torch.nn.SmoothL1Loss(reduction='mean', size_average=False) if opt.kd_loss == 'smooth_l1' else \
                torch.nn.L1Loss(reduction='mean', size_average=False) if opt.kd_loss == 'l1' else None

        self.opt = opt
        self.emb_dim = opt.reid_dim

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, outputs_t, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_output = id_head[batch['reg_mask'] > 0].contiguous()

                if outputs_t == None:
                    id_target = batch['ids'][batch['reg_mask'] > 0]
                else:
                    id_target = outputs_t

                # print(batch.keys())
                # for k in batch.keys():
                #     print(k, ': ', batch[k].shape)
                id_loss += self.crit_kd(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats

class ModelWithLoss_KD(torch.nn.Module):
    def __init__(self, model, loss, teacher_model=None):
        super(ModelWithLoss_KD, self).__init__()
        self.model = model
        self.loss_t = loss
        self.teacher_model = teacher_model

    def forward(self, batch):
        outputs = self.model(batch['input'])

        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.eval()
            with torch.no_grad():
                outputs_t = []
                input_t = batch['cropped_imgs'][batch['reg_mask'] > 0]  # .contiguous()

                # for i in range(batch['cropped_imgs'].shape[0]):
                outputs_t = self.teacher_model.forward(input_t)

        else:
            outputs_t = None

        loss, loss_state = self.loss_t(outputs, outputs_t, batch)
        return outputs[-1], loss, loss_state


class MotKDTrainer(BaseTrainer):
    def __init__(self, opt, model, teacher_model, optimizer=None):
        super(MotKDTrainer, self).__init__(opt, model, optimizer=optimizer)
        self.model_with_loss = ModelWithLoss_KD(model, self.loss, teacher_model)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotKDLoss(opt)
        return loss_states, loss

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results