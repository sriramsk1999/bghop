import json
import os.path as osp
import pickle
from functools import partial

import torch
from jutils import geom_utils, hand_utils
import numpy as np
from scipy.spatial.transform import Rotation as Rt

from .data_utils import get_nXyz_sdf


# NOTE: Values should be the same as in preprocess/arctic_articulated_meshes.py
ARTICULATION_RANGE = np.linspace(0, 1.5*np.pi, 200)

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

    sdf_dir = "arctic_sdf"
    data_dir = "/data/sriram/arctic/"
    object_seq_name  = f"{data_dir}/data/arctic_data/data/raw_seqs/s05/box_use_01.object.npy"
    seq_idx = 150 # random idx ...

    meta['hA'], meta['hTo'], meta['sdf_file'] = get_anno(object_seq_name, seq_idx, sdf_dir)

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

def get_anno(object_seq_name, seq_idx, sdf_dir):
    # Since its an articulated object, we don't have a single SDF for the object
    # Rather than dynamically articulate the object and compute the sdf (slow),
    # we precompute the sdfs for a range of articulations with preprocess/arctic_articulated_meshes.py
    obj_type = osp.basename(object_seq_name).split('_')[0]
    obj_arti = np.load(object_seq_name)[seq_idx, 0]
    closest_articulation = ARTICULATION_RANGE[np.argmin(np.abs(ARTICULATION_RANGE - obj_arti))]
    closest_sdf_file = f"{sdf_dir}/{obj_type}_{closest_articulation:.2f}.npz"
    # closest_sdf_file = "data/arctic_proc/overfit_sdf.npz"

    hand_seq_name = object_seq_name.replace(".object.", ".mano.")
    hA, cTh = get_hand_pose(hand_seq_name, seq_idx)
    cTo = get_cTo(object_seq_name, seq_idx)
    hTo = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTo

    return hA, hTo, [closest_sdf_file]

def get_hand_pose(hand_seq_name, seq_idx):
    device = 'cpu'

    seq = np.load(hand_seq_name, allow_pickle=True).item()

    # NOTE: Only right hand for now.
    seq = seq['right']

    trans = seq['trans'][seq_idx]
    rot = seq['rot'][seq_idx]
    hA = seq['pose'][seq_idx]
    hA = torch.FloatTensor(hA, ).to(device)[None]
    rot = torch.FloatTensor(rot, ).to(device)[None]
    trans = torch.FloatTensor(trans, ).to(device)[None]

    rot, trans = hand_utils.cvt_axisang_t_i2o(rot, trans)
    cTh = geom_utils.axis_angle_t_to_matrix(rot, trans)
    return hA, cTh


def get_cTo(object_seq_name, seq_idx):
    """
    :param index: _description_
    :return: (1, 45)
    """
    obj_poses = np.load(object_seq_name)

    # obj_arti = obj_poses[:, 0, None]  # radian
    obj_rot = obj_poses[:, 1:4]
    obj_trans = obj_poses[:, 4:]

    trans = obj_trans[seq_idx] / 1000. #mm to meters
    rot = obj_rot[seq_idx]
    rt = Rt.from_rotvec(rot).as_matrix()

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
