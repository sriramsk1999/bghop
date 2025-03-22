import json
import os.path as osp
import pickle
from functools import partial

import torch
from jutils import geom_utils, hand_utils
import numpy as np


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

    meta['hA'] = torch.randn(1, 45)
    oTh = torch.eye(4).unsqueeze(0)
    meta['hTo'] = geom_utils.inverse_rt(mat=oTh, return_mat=True)

    if not osp.exists("dummy_sdf.npz"):
        sdf = np.zeros((64, 64, 64))
        transformation = np.eye(4)
        np.savez("dummy_sdf.npz", sdf=sdf, transformation=transformation)


    meta['sdf_file'] = ["dummy_sdf.npz"]

    text_list = ['an image of a hand grasping a dummy object']

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

def get_anno_fast(ind, meta):
    return meta['hA'][ind][None], meta['hTo'][ind]

def get_anno_pred_fast(ind, meta):
    if len(meta['hA_pred']) == 0:
        return None, None
    return meta['hA_pred'][ind][None], meta['hTo_pred'][ind]
