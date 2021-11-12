from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.pose_dla_conv import get_pose_net as get_dla_conv

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dlaconv': get_dla_conv,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path,   map_location=lambda storage, loc: storage)
  print('loaded pretrained model {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  param_mapping_dict = {
          'dla_up.ida_0.proj_1.conv.weight',
          'dla_up.ida_0.node_1.conv.weight',
          'dla_up.ida_1.proj_1.conv.weight',
          'dla_up.ida_1.node_1.conv.weight',
          'dla_up.ida_1.proj_2.conv.weight',
          'dla_up.ida_1.node_2.conv.weight',
          'dla_up.ida_2.proj_1.conv.weight',
          'dla_up.ida_2.node_1.conv.weight',
          'dla_up.ida_2.proj_2.conv.weight',
          'dla_up.ida_2.node_2.conv.weight',
          'dla_up.ida_2.proj_3.conv.weight',
          'dla_up.ida_2.node_3.conv.weight',
          'ida_up.proj_1.conv.weight',
          'ida_up.node_1.conv.weight',
          'ida_up.proj_2.conv.weight',
          'ida_up.node_2.conv.weight',

          'dla_up.ida_0.proj_1.conv.bias',
          'dla_up.ida_0.node_1.conv.bias',
          'dla_up.ida_1.proj_1.conv.bias',
          'dla_up.ida_1.node_1.conv.bias',
          'dla_up.ida_1.proj_2.conv.bias',
          'dla_up.ida_1.node_2.conv.bias',
          'dla_up.ida_2.proj_1.conv.bias',
          'dla_up.ida_2.node_1.conv.bias',
          'dla_up.ida_2.proj_2.conv.bias',
          'dla_up.ida_2.node_2.conv.bias',
          'dla_up.ida_2.proj_3.conv.bias',
          'dla_up.ida_2.node_3.conv.bias',
          'ida_up.proj_1.conv.bias',
          'ida_up.node_1.conv.bias',
          'ida_up.proj_2.conv.bias',
          'ida_up.node_2.conv.bias',
          }

  # print('pretrained')
  # data = open("pretrained.txt", 'w', encoding="utf-8")
  # print(state_dict.keys(), file=data)
  #
  # print('model')
  # data2 = open("model.txt", 'w', encoding="utf-8")
  # print(model_state_dict.keys(), file=data2)

  for k in state_dict.keys():
    if k in param_mapping_dict:
      k_ = k.replace('.conv', '.conv.dcnv2')
      state_dict[k_] = state_dict.pop(k)


  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('Param {} exists in checkpoints while not in new model now.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

