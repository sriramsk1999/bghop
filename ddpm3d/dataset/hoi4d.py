# TODO: clear file
from scipy.spatial.transform import Rotation as Rt
import pickle
from tqdm import tqdm
import numpy as np
import pandas
import torch
import os.path as osp
from functools import partial
from .data_utils import make_cache_fast, get_nXyz_sdf
from jutils import geom_utils, hand_utils

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
    rigid = [
        'Bowl', 'Bottle', 'Mug', 'ToyCar', 'Knife', 'Kettle',
    ]
    meta = {}
    meta['cad_index'] = []
    meta['mesh_file'] = []
    meta['sdf_file'] = []
    meta['hand_file'] = []
    meta['hand_pred_file']  = []
    meta['objpose_file'] = []

    meta['hA'] = []
    meta['hTo'] = []
    meta['hA_pred'] = []
    meta['hTo_pred'] = []
    meta['uTo'] = {}
    meta['uSdf']  = {}

    hand_wrapper = hand_utils.ManopthWrapper().to('cpu')
    hand_mean = hand_wrapper.hand_mean

    set_dir =  osp.join(data_dir, f'Sets/{split}.csv')
    df = pandas.read_csv(set_dir)
    # filter if class is in rigid
    df = df[df['class'].isin(rigid)]
    sdf_folder = args.get('sdf_suf', 'all')

    image_list = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='preload anno'):
        index = row['vid_index'], row['frame_number']
        obj_id = int(row['vid_index'].split('/')[3][1:])
        cad_index = row['class'], obj_id
        image_list.append(index)
        meta['cad_index'].append(cad_index)
        meta['mesh_file'].append(osp.join(data_dir, 'HOI4D_CAD_Model_for_release/rigid/{}/{:03d}.obj').format(*cad_index))
        meta['sdf_file'].append(osp.join(data_dir, 'mesh_sdf/SdfGrids/hoi4d/{:s}/{:s}/{:03d}.npz').format(sdf_folder, *cad_index))
        meta['hand_file'].append(osp.join(data_dir, 'handpose/refinehandpose_right/{}/{:d}.pickle').format(*index))
        meta['hand_pred_file'].append(osp.join(data_dir, 'det_hand/{}/mocap/{:05d}_prediction_result.pkl').format(*index))
        meta['objpose_file'].append(osp.join(data_dir, 'HOI4D_annotations/{}/objpose/{:d}.json').format(*index))

    cache_file = osp.join(data_dir, f'cache/{split}_{sdf_folder}_meta.pkl')
    print('cache file: ', cache_file)
    if args.get('cache_data', False):
        if osp.exists(cache_file):
            meta = pickle.load(open(cache_file, 'rb'))
        else:
            make_cache_fast(args, image_list, meta, cache_file, 
                            get_anno_fn=get_anno, 
                            get_anno_pred_fn=partial(get_anno_pred, hand_mean=hand_mean))

    meta['cat_list'] = []
    mapping = [
        '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
        'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
        'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
    ]
    text_list = []
    for i in range(len(image_list)):
        index = image_list[i][0]
        c = mapping[int(index.split('/')[2][1:])].lower()
        text_list.append(f'an image of a hand grasping a {c}')
        meta['cat_list'].append(c)
    
    print('text set #', len(set(text_list)))
    meta['cfg'] = args
    meta['hand_wrapper'] = hand_utils.ManopthWrapper().to('cpu')

    print('parsing data done!')

    return {
        'image': image_list,
        'text': text_list,
        'img_func': partial(get_nXyz_sdf, get_anno_fn=get_anno), 
        'get_anno_fn': get_anno_fast,
        'get_anno_pred_fn': get_anno_pred_fast,
        'meta': meta,
    }    


def get_hand_pose(right_bbox_dir, pred_file=None):
    device = 'cpu'
    if osp.exists(right_bbox_dir):
        with open(right_bbox_dir, 'rb') as fp:
            obj = pickle.load(fp)
        pose = obj['poseCoeff']
    else:
        raise FileNotFoundError(right_bbox_dir)

    if pred_file is not None and osp.exists(pred_file):
        with open(pred_file, 'rb') as fp:
            obj_pred = pickle.load(fp)
        pose = obj_pred['pred_output_list'][0]['right_hand']['pred_hand_pose'][0]  # (48, )

    beta = obj['beta']
    trans = obj['trans']
    rot, hA = pose[:3], pose[3:]
    hA = torch.FloatTensor(hA, ).to(device)[None]
    rot = torch.FloatTensor(rot, ).to(device)[None]
    trans = torch.FloatTensor(trans, ).to(device)[None]
    beta = torch.FloatTensor(beta, ).to(device)[None]
    
    rot, trans = hand_utils.cvt_axisang_t_i2o(rot, trans)
    cTw = geom_utils.axis_angle_t_to_matrix(rot, trans)
    pose = torch.FloatTensor(pose[None]).to(device)

    return hA, cTw


def get_cTo(pose_file):
    """
    :param index: _description_
    :return: (1, 45)
    """
    rt, trans, dim = read_rtd(pose_file, 0)
    rt = torch.FloatTensor(rt)[None]
    trans = torch.FloatTensor(trans)[None]
    cTo = geom_utils.rt_to_homo(rt, trans)
    return cTo


def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
        cont = eval(cont)
    if "dataList" in cont:
        anno = cont["dataList"][num]
    else:
        anno = cont["objects"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_matrix()
    return np.array(rot, dtype=np.float32), trans, dim


def get_anno(ind, meta):
    hand_file, objpose_file = meta['hand_file'][ind], meta['objpose_file'][ind]
    hA, cTh = get_hand_pose(hand_file)
    cTo = get_cTo(objpose_file)
    hTo = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTo
    return hA, hTo


def get_anno_pred(ind, meta, hand_mean):
    hand_pred_file, hand_file, objpose_file = meta['hand_pred_file'][ind], meta['hand_file'][ind], meta['objpose_file'][ind]
    hA, cTh = get_hand_pose(hand_file, pred_file=hand_pred_file)
    hA = hA + hand_mean
    cTo = get_cTo(objpose_file)
    hTo = geom_utils.inverse_rt(mat=cTh, return_mat=True) @ cTo
    return hA, hTo

def get_anno_fast(ind, meta,):
    if meta['cfg'].cache_data:
        hA_gt, hTo_gt = meta['hA'][ind], meta['hTo'][ind]
        hA_gt = torch.FloatTensor(hA_gt)
        hTo_gt = torch.FloatTensor(hTo_gt)
    else:
        hA_gt, hTo_gt = get_anno(ind, meta)    

    return hA_gt, hTo_gt


def get_anno_pred_fast(ind, meta):
    if meta['cfg'].cache_data:
        hA_pred, hTo_pred = meta['hA_pred'][ind], meta['hTo_pred'][ind]
        hA_pred = torch.FloatTensor(hA_pred)
        hTo_pred = torch.FloatTensor(hTo_pred)
    else:
        hA_pred, hTo_pred = get_anno_pred(ind, meta, meta['hand_wrapper'].hand_mean)
    return hA_pred, hTo_pred
