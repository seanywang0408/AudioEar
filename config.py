'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')

# ---------------------------------------------------------------------------- #
# Options for ear model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.hera_ear_path = './data/deca/hera_ear_large.obj'

## pickel file containing model's mean, engienvectors, engienvalues
cfg.model.pkl_path = './data/UHM_models/ear_model.pkl' 

## xlsx file containing model's landmark index
cfg.model.land_mark_path = './data/55_landmarks.xlsx'

## path to dataset pictures

cfg.model.ear_dataset_path = './data/AudioEar2D'
cfg.model.sythetic_dataset_path = './data/sythc_data'
cfg.model.tex_path = './data/FLAME_albedo_from_BFM.npz'
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.depth_model_path = ''
cfg.model.uv_size = 256
cfg.model.n_tex = 50
cfg.model.img_size = 256
## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Training
cfg.train = CN()
cfg.train.batch_size = 8
cfg.train.pos_epoch = -1
cfg.train.shape_epoch = 30
cfg.train.texure_epoch = 30
cfg.train.total_epoch = 100

## whether using independent resnet encoder for the shape and position
cfg.train.two_res = False

# ---------------------------------------------------------------------------- #
# Options for Dataset
cfg.data = CN()
# width of the padding pixels around ear
cfg.data.img_pad = 100

# ---------------------------------------------------------------------------- #
# Options for Scan2mesh test
cfg.s2m = CN()
## path to front face obj
cfg.s2m.ear_face_model_path = './data/s2m_ear_model/ear_model_front.obj'
cfg.s2m.s2m_data_path = './data/AudioEar3D'
cfg.s2m.recon_model_path = ''
## number of points sampled from groundtruth
cfg.s2m.pts_num = 1000
cfg.s2m.total_epoch = 500
cfg.s2m.keypoint_epoch = 2000
cfg.s2m.mix_epoch = 7000
cfg.s2m.recon_model_path =  ''
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
