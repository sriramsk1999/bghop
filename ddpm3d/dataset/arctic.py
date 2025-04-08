import json
import os.path as osp
import pickle
from functools import partial

import torch
from jutils import geom_utils, hand_utils
import numpy as np
from scipy.spatial.transform import Rotation as Rt
import pandas as pd
from tqdm import tqdm

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
    meta['hA_left'] = []
    meta['hTh_left'] = []
    meta['hA_pred'] = []
    meta['hTo_pred'] = []
    meta['sdf_file'] = []
    meta['fname'] = []

    meta['uTo'] = {}
    meta['uSdf']  = {}

    meta['cfg'] = args
    meta['hand_wrapper'] = hand_utils.ManopthWrapper(args.environment.mano_dir, flat_hand_mean=data_cfg.flat_hand_mean).to('cpu')

    text_list = []
    seq_prefix = "arctic_data/data/raw_seqs"

    df = pd.read_csv(osp.join(data_dir, f'arctic_contact_{split}.csv'))
    for i, row in tqdm(df.iterrows(), total=len(df), desc='preload anno'):
        seq_name, seq_idx, caption = row['filename'], row['timestep'], row['caption']

        hand_seq_name  = f"{data_dir}/{seq_prefix}/{seq_name}"
        object_seq_name  = hand_seq_name.replace('.mano.', '.object.')

        meta['sdf_file'].append(get_closest_sdf_file(data_dir, object_seq_name, seq_idx))
        hand_info = get_anno(hand_seq_name, object_seq_name, seq_idx)
        meta['hA'].append(hand_info["right"][0])
        meta['hTo'].append(hand_info["right"][1])
        meta['hA_left'].append(hand_info["left"][0])
        meta['hTh_left'].append(hand_info["left"][1])
        meta['fname'].append(f"{seq_name}-{seq_idx}")
        text_list.append(caption)

    return {
        'image': meta['sdf_file'],
        'text': text_list,
        'img_func': partial(get_nXyz_sdf, get_anno_fn=get_anno_fast),
        'get_anno_fn': get_anno_fast,
        'get_anno_pred_fn': get_anno_pred_fast,
        'meta': meta,
    }

def get_anno(hand_seq_name, object_seq_name, seq_idx):
    hand_anno = get_hand_pose(hand_seq_name, seq_idx)
    hA, cTh = hand_anno["right"]
    hA_left, cTh_left = hand_anno["left"]
    cTo = get_cTo(object_seq_name, seq_idx)
    hTo = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTo
    hTh_left = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTh_left

    return {"left": (hA_left, hTh_left), "right": (hA, hTo)}

def get_hand_pose(hand_seq_name, seq_idx):
    device = 'cpu'

    hand_anno = {}
    seq = np.load(hand_seq_name, allow_pickle=True).item()

    for key in ["left", "right"]:
        hand_seq = seq[key]

        trans = hand_seq['trans'][seq_idx]
        rot = hand_seq['rot'][seq_idx]
        hA = hand_seq['pose'][seq_idx]
        hA = torch.FloatTensor(hA, ).to(device)
        rot = torch.FloatTensor(rot, ).to(device)[None]
        trans = torch.FloatTensor(trans, ).to(device)[None]

        rot, trans = hand_utils.cvt_axisang_t_i2o(rot, trans)
        cTh = geom_utils.axis_angle_t_to_matrix(rot, trans)
        hand_anno[key] = (hA, cTh)
    return hand_anno


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

def get_closest_sdf_file(data_dir, object_seq_name, seq_idx):
    # Since its an articulated object, we don't have a single SDF for the object
    # Rather than dynamically articulate the object and compute the sdf (slow),
    # we precompute the sdfs for a range of articulations with preprocess/arctic_articulated_meshes.py
    # This means we won't have the SDF for the *exact* articulation, but we'll be within 1-2 degrees.

    obj_type = osp.basename(object_seq_name).split('_')[0]
    obj_arti = np.load(object_seq_name)[seq_idx, 0] # first element is articulation
    closest_articulation = ARTICULATION_RANGE[np.argmin(np.abs(ARTICULATION_RANGE - obj_arti))]
    closest_sdf_file = f"{data_dir}/arctic_sdf/{obj_type}_{closest_articulation:.2f}.npz"
    return closest_sdf_file

def get_anno_fast(ind, meta):
    return meta['hA'][ind][None], meta['hTo'][ind], meta['hA_left'][ind][None], meta['hTh_left'][ind]

def get_anno_pred_fast(ind, meta):
    if len(meta['hA_pred']) == 0:
        return None, None
    return meta['hA_pred'][ind][None], meta['hTo_pred'][ind]
