# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os
import os.path as osp
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from jutils import geom_utils, image_utils, mesh_utils, plot_utils
from tqdm import tqdm

from utils.io_util import glob_imgs, load_mask, load_rgb


class SceneDataset(torch.utils.data.Dataset):
    # NOTE: for HOI
    # modified from IDR.   https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 len_data=1000,
                 suf='',
                 enable_bimanual=False,
                 args=dict(), **kwargs):

        self.enable_bimanual = enable_bimanual
        assert os.path.exists(data_dir), "Data directory is empty %s" % data_dir
        if osp.exists(osp.join(data_dir, 'oObj.obj')):
            self.oObj = mesh_utils.load_mesh(osp.join(data_dir, 'oObj.obj'))
        else:
            self.oObj = plot_utils.create_coord('cpu', 1)
        self.instance_dir = data_dir
        self.train_cameras = train_cameras
        if osp.exists(osp.join(data_dir, 'text.txt')):
            cat = open(osp.join(data_dir, 'text.txt')).read().strip().lower()
        else:
            cat = 'object'
        # self.text = f'an image of a hand grasping a {cat}'
        self.text = f'{cat}'
        print(self.text)
        self.suf = suf

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(glob_imgs(mask_dir))
        self.n_images = min(len(image_paths), len_data)
                
        print('len image', len(self))
        # determine width, height
        self.downscale = downscale
        tmp_rgb = load_rgb(image_paths[0], downscale)
        _, self.H, self.W = tmp_rgb.shape
        # load camera and pose
        self.cam_file = '{0}/cameras_hoi{1}.npz'.format(self.instance_dir, self.suf)
        camera_dict = np.load(self.cam_file)
        self.wTc = geom_utils.inverse_rt(mat=torch.from_numpy(camera_dict['cTw']).float(), return_mat=True)
        N = len(self.wTc)
        idty = torch.eye(4)[None].repeat(N, 1, 1)
        if 'wTh' in camera_dict:
            self.wTh = torch.from_numpy(camera_dict['wTh']).float()  # a scaling mat that recenter hoi to -1, 1? 
        else:
            self.wTh = idty
        self.hTo = camera_dict.get('hTo', idty)
        scale = idty.clone()
        scale[:, 0, 0] = scale[:, 1, 1] = scale[:, 2, 2] = 10
        self.onTo = camera_dict.get('onTo', scale)
        self.intrinsics_all = torch.from_numpy(camera_dict['K_pix']).float()  # (N, 4, 4)
        # downscale * self.H is the orignal height before resize. 
        self.intrinsics_all = mesh_utils.intr_from_screen_to_ndc(
            self.intrinsics_all, downscale* self.H, downscale * self.W)

        self.scale_cam = 1
        # calculate cam distance
        cam_center_norms = []
        for wTc in self.wTc:
            cam_center_norms.append(np.linalg.norm(wTc[:3,3].detach().numpy()))
        max_cam_norm = max(cam_center_norms)
        self.max_cam_norm = max_cam_norm  # in camera metric??? 
        print('max cam norm', self.max_cam_norm)

        # TODO: crop??!!!
        self.rgb_images = []
        for path in tqdm(image_paths, desc='loading images...'):
            rgb = load_rgb(path, downscale)
            H, W = rgb.shape[1:]
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        # load hand 
        hands = np.load('{0}/hands{1}.npz'.format(self.instance_dir, self.suf))
        self.hA = torch.from_numpy(hands['hA']).float().squeeze(1)
        self.beta = torch.from_numpy(hands['beta']).float().squeeze(1).mean(0)
        self.hA_inp = self.hA

        image_dir = '{0}/obj_mask'.format(self.instance_dir)
        obj_paths = sorted(glob_imgs(image_dir))
        image_dir = '{0}/hand_mask'.format(self.instance_dir)
        hand_paths = sorted(glob_imgs(image_dir))

        self.obj_masks = []
        for path in obj_paths:
            object_mask = load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.obj_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        self.hand_masks = []
        self.hand_contours = []
        for path in hand_paths:
            object_mask = load_mask(path, downscale)
            hand_c = image_utils.sample_contour(object_mask > 0).astype(np.float32) / object_mask.shape[0] * 2 - 1
            self.hand_contours.append(hand_c)
            object_mask = object_mask.reshape(-1)
            self.hand_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        if self.enable_bimanual:
            self.hA_left = torch.from_numpy(hands['hA_left']).float().squeeze(1)
            self.beta_left = torch.from_numpy(hands['beta_left']).float().squeeze(1)
            self.hA_left_inp = self.hA_left
            image_dir = '{0}/hand_left_mask'.format(self.instance_dir)
            hand_left_paths = sorted(glob_imgs(image_dir))
            self.hand_left_masks = []
            self.hand_left_contours = []
            for path in hand_left_paths:
                object_mask = load_mask(path, downscale)
                hand_c = image_utils.sample_contour(object_mask > 0).astype(np.float32) / object_mask.shape[0] * 2 - 1
                self.hand_left_contours.append(hand_c)
                object_mask = object_mask.reshape(-1)
                self.hand_left_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        if len(self.object_masks) == 0:
            for obj_mask, hand_mask in zip(self.obj_masks, self.hand_masks):
                self.object_masks.append(torch.logical_or(obj_mask, hand_mask))

        depth_paths = sorted(glob(os.path.join(self.instance_dir, 'depth', '*_depth.npy')))
        self.depth_images = []
        for dpath in depth_paths:
            depth = torch.from_numpy(np.load(dpath))  # ()
            # resize to (H, W)
            depth = TF.resize(depth[None], (H, W))
            self.depth_images.append(depth.reshape(-1).float())
        if len(depth_paths) == 0:
            self.depth_images = torch.zeros([len(self.rgb_images), H * W])
        
        normal_paths = sorted(glob(os.path.join(self.instance_dir, 'normal', '*_normal.npy')))
        self.normal_images = []

        for npath in normal_paths:
            normal = torch.from_numpy(np.load(npath))
            normal = TF.resize(normal, (H, W))
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.

            x,y,z=normal[:,0],normal[:,1],normal[:,2]
            normal = torch.stack([x, -y, -z],dim=1)
            # normalize
            normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
            self.normal_images.append(normal.float())
        if len(normal_paths) == 0:
            self.normal_images = torch.ones([len(self.rgb_images), H * W, 3])

    def __len__(self):
        return self.n_images - 1

    def __getitem__(self, idx): 
        # TODO: support crop!
        idx_n = idx + 1
        sample = {
            "object_mask": self.object_masks[idx],
            "intrinsics":  self.intrinsics_all[idx],
            "intrinsics_n": self.intrinsics_all[idx_n],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        ground_truth["rgb"] = self.rgb_images[idx]
        ground_truth['normal'] = self.normal_images[idx]
        ground_truth['depth'] = self.depth_images[idx]

        sample["object_mask"] = self.object_masks[idx]

        sample['inds_n'] = idx_n
        
        if not self.train_cameras:
            sample["c2w"] = self.wTc[idx]
            sample['c2w_n'] = self.wTc[idx_n]

            sample['wTh'] = self.wTh[idx]
            sample['wTh_n'] = self.wTh[idx_n]

            sample['hTo'] = self.hTo[idx]
            sample['hTo_n'] = self.hTo[idx_n]
            
            sample['onTo'] = self.onTo[idx]
            sample['onTo_n'] = self.onTo[idx_n]

        sample['obj_mask'] = self.obj_masks[idx]
        sample['hand_mask'] = self.hand_masks[idx]
        sample['hand_contour'] = self.hand_contours[idx]
        sample['hA'] = self.hA_inp[idx]
        sample['hA_n'] = self.hA_inp[idx_n]
        sample['th_betas'] = self.beta

        ground_truth['hA'] = self.hA[idx]
        sample['text'] = self.text

        if self.enable_bimanual:
            sample['hand_left_mask'] = self.hand_left_masks[idx]
            sample['hand_left_contour'] = self.hand_left_contours[idx]

            sample['hA_left'] = self.hA_left_inp[idx]
            sample['hA_left_n'] = self.hA_left_inp[idx_n]
            sample['th_betas_left'] = self.beta_left

            ground_truth['hA_left'] = self.hA_left[idx]

        idx_n = idx + 1
        sample['inds_n'] = idx_n
        return idx, sample, ground_truth

if __name__ == "__main__":
    print('Hello')
