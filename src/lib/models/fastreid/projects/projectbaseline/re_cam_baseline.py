# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from projects.centerTrack.fastreid.modeling.losses import *
from projects.centerTrack.fastreid.modeling.meta_arch import Baseline
from projects.centerTrack.fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from projects.centerTrack.fastreid.modeling.backbones import build_backbone
from projects.centerTrack.fastreid.modeling.heads import build_reid_heads

import copy

@META_ARCH_REGISTRY.register()
class ReCamBaseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        backbone = build_backbone(cfg)
        
        # head
        self.heads = build_reid_heads(cfg)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
            )
        self.id_branch = copy.deepcopy(backbone.layer4)
        self.cam_branch = copy.deepcopy(backbone.layer4)


    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        print("re_cam_baseline", images)
        layer3_features = self.backbone(images)  # (bs, 2048, 16, 8)
        #id_feat = self.id_backbone(images)  # (bs, 2048, 16, 8)
        #cam_feat = self.cam_backbone(images)  # (bs, 2048, 16, 8)

        # id_branch1
        id_feat = self.id_branch(layer3_features)

        # branch2
        cam_feat = self.cam_branch(layer3_features)

        features = id_feat - 0.15 * cam_feat

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            camids = batched_inputs["camids"].long().to(self.device)
            #outputs["camids"] = camids

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, cam_feat, targets)
            return {
                "outputs": outputs,
                "targets": targets,
                "camids": camids,
            }
        else:
            outputs = self.heads(features)
            return outputs


    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        # return images
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        cam_labels      = outs["camids"]
        cam_cls_outputs = outputs["camid_cls_outputs"]
        loss_dict['loss_cam'] = cross_entropy_loss(cam_cls_outputs, cam_labels, 0.1) * 0.15

        return loss_dict

