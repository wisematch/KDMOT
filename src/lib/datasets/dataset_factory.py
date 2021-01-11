from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset, CropJointDataset


def get_dataset(dataset, task):
  if task == 'mot':
    return JointDataset
  elif task == 'kd':
    return CropJointDataset
  else:
    return None
  
