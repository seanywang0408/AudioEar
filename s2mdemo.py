
import os
import time
import pickle

from pytorch3d.io.obj_io import load_obj
import _init_paths

from backup_utils import backup_code, backup_terminal_outputs, set_seed
set_seed(1000)


import torch
import torch.nn as nn
import numpy as np
import open3d


import pandas as pd

from pytorch3d.io import load_ply, save_obj, save_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance, mesh_normal_consistency
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.renderer import look_at_rotation
from pytorch3d.ops import sample_points_from_meshes
from scipy.spatial import ConvexHull
from custom_loss import sample_points_from_polylines, custom_chamfer_distance
from dataset import EarValLoader
from models import ResNet_concat, ResNet_FCRN
from torch.autograd import Variable
from config import cfg
from s2mtest import s2m_test
import FCRN

shape_vec_value = 'pred'
assert shape_vec_value == 'pred' or shape_vec_value == 'avg' or shape_vec_value == 'optim'


if shape_vec_value == 'optim':
    save_path = os.path.join('./log/s2m/', time.strftime("%y%m%d_%H%M%S"+'_optimize_shape'))
elif shape_vec_value == 'avg':
    save_path = os.path.join('./log/s2m/', time.strftime("%y%m%d_%H%M%S"+'_avg_ear'))
else:
    save_path = os.path.join('./log/s2m/', time.strftime("%y%m%d_%H%M%S"+''))

os.makedirs(save_path, exist_ok=True)
print('save_path', save_path)
backup_code(save_path)
backup_terminal_outputs(save_path)


s2m = s2m_test(cfg, save_path, shape_vec_value=shape_vec_value)

fcrn_model = FCRN.ResNet(layers=34, output_size=(256, 256)).cuda()
fcrn_model.load_state_dict(torch.load(cfg.model.depth_model_path))

fcrn_model.eval()

model_path = cfg.s2m.recon_model_path#  210826_092305_recon     

encoder = ResNet_FCRN().cuda()
encoder_state_dict = torch.load(model_path)
encoder.load_state_dict(encoder_state_dict)
encoder.eval()       
s2m.test(encoder,fcrn_model)