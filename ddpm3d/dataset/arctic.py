import json
import os.path as osp
import pickle
from functools import partial

import torch
from jutils import geom_utils, hand_utils
import numpy as np
from scipy.spatial.transform import Rotation as Rt

from .data_utils import get_nXyz_sdf


def parse_data(data_dir, split, data_cfg, args):
    """_summary_

    :param data_dir: _description_
    :param split: _description_
    :param data_cfg: _description_
    :param args: _description_
    :return: format according to base_data.py
        'image': list of HOI files
        'text': list of str
        'img_func':
        'meta': {'nTo', 'hA'}
    """
    meta = {}

    meta['hA'] = []
    meta['hTo'] = []
    meta['hA_pred'] = []
    meta['hTo_pred'] = []
    meta['sdf_file'] = []

    meta['uTo'] = {}
    meta['uSdf']  = {}

    ROOT_DIR = "/data/sriram/arctic/"
    seq_name  = f"{ROOT_DIR}/outputs/processed_verts/seqs/s05/box_use_01.npy"
    seq_idx = 150 # random idx ...

    meta['hA'], meta['hTo'] = get_anno(seq_name, seq_idx)

    meta['sdf_file'] = ["data/arctic_proc/overfit_sdf.npz"]
    text_list = ['an image of a hand grasping a box']

    meta['cfg'] = args
    meta['cad_index'] = [0]
    meta['hand_wrapper'] = hand_utils.ManopthWrapper(args.environment.mano_dir).to('cpu')

    return {
        'image': meta['sdf_file'],
        'text': text_list,
        'img_func': partial(get_nXyz_sdf, get_anno_fn=get_anno_fast),
        'get_anno_fn': get_anno_fast,
        'get_anno_pred_fn': get_anno_pred_fast,
        'meta': meta,
    }

def get_anno(seq_name, seq_idx):
    seq = np.load(seq_name, allow_pickle=True).item()
    hA, cTh = get_hand_pose(seq, seq_idx)
    cTo = get_cTo(seq, seq_idx)
    hTo = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTo
    return hA, hTo

def get_hand_pose(seq, seq_idx):
    device = 'cpu'

    trans = seq["params"]['trans_r'][seq_idx]
    rot = seq["params"]['rot_r'][seq_idx]
    hA = seq["params"]['pose_r'][seq_idx]
    hA = torch.FloatTensor(hA, ).to(device)[None]
    rot = torch.FloatTensor(rot, ).to(device)[None]
    trans = torch.FloatTensor(trans, ).to(device)[None]

    rot, trans = hand_utils.cvt_axisang_t_i2o(rot, trans)
    cTh = geom_utils.axis_angle_t_to_matrix(rot, trans)
    return hA, cTh


def get_cTo(seq, seq_idx):
    """
    :param index: _description_
    :return: (1, 45)
    """
    trans = seq["params"]['obj_trans'][seq_idx]

    rot = seq["params"]['obj_rot'][seq_idx]
    rt = Rt.from_euler('XYZ', rot).as_matrix()

    rt = torch.FloatTensor(rt)[None]
    trans = torch.FloatTensor(trans)[None]
    cTo = geom_utils.rt_to_homo(rt, trans)
    return cTo

def get_anno_fast(ind, meta):
    return meta['hA'][ind][None], meta['hTo'][ind]

def get_anno_pred_fast(ind, meta):
    if len(meta['hA_pred']) == 0:
        return None, None
    return meta['hA_pred'][ind][None], meta['hTo_pred'][ind]
