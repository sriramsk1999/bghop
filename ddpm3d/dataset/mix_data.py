# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import json
import os.path as osp
import pickle
from functools import partial

import torch
from jutils import geom_utils, hand_utils

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
    print('parsing data...')
    meta = {}
    
    meta['hA'] = []
    meta['hTo'] = []
    meta['hA_pred'] = []
    meta['hTo_pred'] = []
    meta['sdf_file'] = []

    meta['uTo'] = {}
    meta['uSdf']  = {}
     
    anno_list = pickle.load(open(osp.join(data_dir, data_cfg.anno_file), 'rb'))
    
    meta['hA'] = torch.FloatTensor(anno_list['hA'])
    oTh = torch.FloatTensor(anno_list['oTh'])
    meta['hTo'] = geom_utils.inverse_rt(mat=oTh, return_mat=True)

    if 'hA_pred' in anno_list:
        meta['hA_pred'] = torch.FloatTensor(anno_list['hA_pred'])
        oTh_pred = torch.FloatTensor(anno_list['oTh_pred'])
        meta['hTo_pred'] = geom_utils.inverse_rt(mat=oTh_pred, return_mat=True)
    meta['sdf_file'] = anno_list['sdf']

    # get categories
    if osp.exists(data_cfg.map_file):
        obj2cat_file = json.load(open(data_cfg.map_file, 'r'))
        meta['cat_list'] = [obj2cat_file[obj] for obj in anno_list['obj']]
    else:
        meta['cat_list'] = anno_list['obj']
    
    text_list = []        
    for c in meta['cat_list']:
        text_list.append(f'an image of a hand grasping a {c}')

    if args.get('property', False):
        with open(args.property_file, 'r') as f:
            cat2property = json.load(f)
        for i, c in enumerate(meta['cat_list']):
            p = ', '.join(cat2property.get(c, []))
            text_list[i] += + ', ' + p
    print('text set #', len(set(text_list)))
    meta['cfg'] = args
    meta['hand_wrapper'] = hand_utils.ManopthWrapper(args.environment.mano_dir).to('cpu')

    print('parsing data done!')

    return {
        'image': meta['sdf_file'],
        'text': text_list,
        'img_func': partial(get_nXyz_sdf, get_anno_fn=get_anno_fast), 
        # 'cond_func': get_anno_fast,
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