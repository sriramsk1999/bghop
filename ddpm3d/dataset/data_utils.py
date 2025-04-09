# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import multiprocessing
import os
import os.path as osp
import pickle
from copy import deepcopy

import numpy as np
import torch
from jutils import geom_utils, hand_utils, mesh_utils
from tqdm import tqdm


def get_nXyz_sdf(index, ind, meta, get_anno_fn, hA=None, hTo=None, hTh_left=None, **kwargs):
    """
    :return: nXyz:  (D, H, W, 3) just be grid [-lim, lim]
    :return: nSdf:  (D, H, W, ) SDF in normalized hand frame
    """
    # import pdb; pdb.set_trace()
    H = meta['cfg'].side_x
    lim = meta['cfg'].side_lim
    up_sample = meta['cfg'].get('side_up', None)
    hand_wrapper = meta['hand_wrapper']
    device = hand_wrapper.hand_mean.device

    sdf_file = meta['sdf_file'][ind] 
    if meta['cfg'].cache_data:
        cad_index = meta['cad_index'][ind]
        uSdf, uTo = meta['uSdf'][cad_index], meta['uTo'][cad_index]
        uSdf = torch.FloatTensor(uSdf)
        uTo = torch.FloatTensor(uTo)
    else:
        uSdf, uTo = load_sdf_grid(sdf_file, tensor=True, batched=True,)
        uSdf = uSdf.to(device)
        uTo = uTo.to(device)
    
    if hA is None or hTo is None:
        if meta['cfg'].cache_data:
            hA, hTo = meta['hA'][ind], meta['hTo'][ind]
            hA = torch.FloatTensor(hA)
            hTo = torch.FloatTensor(hTo)
        else:
            hA, hTo = get_anno_fn(ind, meta)
    
    nTh = hand_utils.get_nTh(hand_wrapper=hand_wrapper, hA=hA)

    nTo = nTh @ hTo  
    oTu = geom_utils.inverse_rt(mat=uTo, return_mat=True)
    nTu = nTo @ oTu

    if hTh_left is not None:
        nTh_left = nTh @ hTh_left
        # Scales the translation according to nTh
        # Without this, the translation is too large for some reason.
        # With this, its still wrong, but its almost correct.
        # TODO: Debug what's going wrong.
        nTh_left[:,:3,3] = (nTh_left[:,:3,3] / torch.linalg.norm(nTh_left[:,:3,3], dim=1, keepdim=True)) * torch.linalg.norm(nTh[:,:3,3], dim=1, keepdim=True)
    else:
        nTh_left = None

    out = {}
    if not meta['cfg'].gpu_trans:
        nSdf, nXyz = mesh_utils.transform_sdf_grid(uSdf, nTu, N=H, lim=lim, N_up=up_sample, **kwargs)
        out['nSdf'] = nSdf[0]
        out['nXyz'] = nXyz[0]
    else:
        out['uSdf'] = uSdf[0]
        out['nTu'] = nTu[0]

    if meta['cfg'].get('rep', 'sdf') == 'occ':
        out['nSdf'] = (out['nSdf'] < 0).float() * 2 - 1  # [-1, 1]
    return out, nTo, nTh_left


def load_sdf_grid(sdf_file, tensor=False, batched=False, device='cpu'):
    """

    :return: (N, N, N), (4, 4) of numpy
    """
    obj = np.load(sdf_file)
    sdf = obj['sdf']
    transformation = obj['transformation']
    if tensor:
        sdf = torch.FloatTensor(sdf).to(device)[None]
        transformation = torch.FloatTensor(transformation).to(device)
    if batched:
        sdf = sdf[None]
        transformation = transformation[None]
    return sdf, transformation


def process_data(inputs):
    start, end, meta, get_anno = inputs

    hA_list = []
    hTo_list = []
    
    for ind in tqdm(range(start, end), desc='caching data'):
        hA, hTo = get_anno(ind, meta)
        hA_list.append(hA.cpu().numpy())
        hTo_list.append(hTo.cpu().numpy())
    
    return hA_list, hTo_list

def make_cache_fast(cfg, image_list, meta, cache_file, get_anno_fn, k=40, get_anno_pred_fn=None):
    num_images = len(image_list)
    image_list = np.array(image_list)
    if k == 0:
        hA_list, hTo_list = process_data((0, num_images, meta, get_anno_fn))
        meta['hA'] = hA_list
        meta['hTo'] = hTo_list
        if get_anno_pred_fn is not None:
            meta['hA_pred'], meta['hTo_pred']  = process_data((0, num_images, meta, get_anno_pred_fn))
    else:        
        chunk_size = num_images // k

        indices = [(i * chunk_size, (i + 1) * chunk_size) for i in range(k - 1)]
        indices.append(((k - 1) * chunk_size, len(image_list)))

        process = multiprocessing.Pool(processes=k)

        data = [(start, end, deepcopy(meta), get_anno_fn) for start, end in indices]
        results = process.map(process_data, data)

        for sub_results in results:
            hA_list, hTo_list = sub_results
            meta['hA'].extend(hA_list)
            meta['hTo'].extend(hTo_list)

        if get_anno_pred_fn is not None:
            data = [(start, end, deepcopy(meta), get_anno_pred_fn) for start, end in indices]
            results = process.map(process_data, data)

            for sub_results in results:
                hA_list, hTo_list = sub_results
                meta['hA_pred'].extend(hA_list)
                meta['hTo_pred'].extend(hTo_list)
                
    obj_meta = _make_obj_cache(cfg, image_list, meta)
    meta.update(obj_meta)

    os.makedirs(osp.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(meta, f)
        
    return meta


def _make_obj_cache(cfg, image_list, meta):
    cache = {}
    # cache object-centric variable
    uSdf_list = {}
    uTo_list = {}

    for ind in tqdm(range(len(image_list))):
        cad_index = meta['cad_index'][ind]
        if cad_index not in uSdf_list:
            uSdf, uTo = load_sdf_grid(meta['sdf_file'][ind], True, batched=True)
            uSdf_list[cad_index] = uSdf.cpu().numpy()
            uTo_list[cad_index] = uTo.cpu().numpy()
    cache['uSdf'] = uSdf_list
    cache['uTo'] = uTo_list

    return cache
