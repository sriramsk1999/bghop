import os
import os.path as osp
from functools import partial
from glob import glob

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from jutils import (
    geom_utils,
    image_utils,
    mesh_utils,
    model_utils,
    plot_utils,
    slurm_utils,
)
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataio import get_data
from models.cameras import get_camera
from models.frameworks import get_model
from utils import mesh_util
from utils.dist_util import is_master


def run(
    dataloader,
    trainer,
    save_dir,
    name,
    H,
    W,
    offset=None,
    N=64,
    volume_size=6,
    max_t=1000,
):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 2]]).to(device),
        )
    rot_y = geom_utils.axis_angle_t_to_matrix(
        np.pi / 2 * torch.FloatTensor([[1, 0, 0]]).to(device),
    )

    model = trainer.model
    renderer = partial(trainer.mesh_renderer, soft=False)

    os.makedirs(save_dir, exist_ok=True)
    # reconstruct object
    # if is_master():
    mesh_util.extract_mesh(
        model.implicit_surface,
        N=N,
        filepath=osp.join(save_dir, name + "_obj.ply"),
        volume_size=volume_size,
    )
    jObj = mesh_utils.load_mesh(osp.join(save_dir, name + "_obj.ply")).cuda()

    # jTn_list = []
    # reconstruct  hand and render in normazlied frame
    name_list = ["gt", "view_0", "view_j", "view_h", "view_hy", "obj", "jHoi"]
    image_list = [[] for _ in name_list]
    for i, (indices, model_input, ground_truth) in enumerate(dataloader):
        if i >= max_t:
            break
        hh = ww = int(np.sqrt(ground_truth["rgb"].size(1)))
        gt = ground_truth["rgb"].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        gt = F.adaptive_avg_pool2d(gt, (H, W))

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W
        )
        hTj = geom_utils.inverse_rt(mat=jTh, return_mat=True)
        image1, _ = render(renderer, jHand, jObj, jTc, intrinsics, H, W)

        image3, _ = render(renderer, jHand, jObj, None, None, H, W)
        image4, _ = render(
            renderer,
            mesh_utils.apply_transform(jHand, hTj),
            mesh_utils.apply_transform(jObj, hTj),
            None,
            None,
            H,
            W,
        )
        image5, _ = render(
            renderer,
            mesh_utils.apply_transform(jHand, rot_y @ hTj),
            mesh_utils.apply_transform(jObj, rot_y @ hTj),
            None,
            None,
            H,
            W,
        )

        image_list[0].append(gt)
        image_list[1].append(image1)  # view 0
        image_list[2].append(image3)  # view_j
        image_list[3].append(image4)  # view _h
        image_list[4].append(image5)

    image_list[5] = mesh_utils.render_geom_rot(jObj, scale_geom=True)
    jHand.textures = mesh_utils.pad_texture(jHand, "blue")
    jObj.textures = mesh_utils.pad_texture(jObj, "yellow")
    HH = 256
    time_len = 30
    K_ndc = mesh_utils.intr_from_screen_to_ndc(intrinsics, H, W)
    gif_list = []
    for i in range(time_len):
        jTcp = mesh_utils.get_wTcp_in_camera_view(2 * i * np.pi / time_len, wTc=jTc)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(
            jHand, jObj, jTcp, H=HH, K_ndc=K_ndc, bin_size=32
        )
        gif_list.append(hoi)
    image_list[6] = gif_list

    # toto: render novel view!
    if is_master():
        for n, im_list in zip(name_list, image_list):
            image_utils.save_gif(im_list, osp.join(save_dir, name + "_%s" % n))

    file_list = [osp.join(save_dir, name + "_%s.gif" % n) for n in name_list]
    return file_list



@torch.no_grad()
def run_video(dataloader, trainer, save_dir, name, H, W,  N=64, volume_size=6, max_t=1000, T_num=None):
    device = trainer.device
    model = trainer.model
    save_dir = osp.join(save_dir, name, )
    os.makedirs(save_dir, exist_ok=True)
    mesh_file = osp.join(save_dir, 'jObj.ply')
    if not osp.exists(mesh_file):
        try:
            mesh_util.extract_mesh(
                    model.implicit_surface, 
                    # level=0.05,
                    N=N,
                    filepath=mesh_file,
                    volume_size=volume_size,
                )
        except ValueError:
            jObj = plot_utils.create_coord('cpu', size=0.00001)
            mesh_utils.dump_meshes([mesh_file[:-4]], jObj)
            mesh_file = mesh_file.replace('.ply', '.obj')
    jObj = mesh_utils.load_mesh(mesh_file).cuda()
    jObj.textures = mesh_utils.pad_texture(jObj, 'yellow')

    name_list = ['input', 'render_0', 'render_1', 'jHoi', 'jObj', 'vHoi', 'vObj', 'vObj_t', 'vHoi_fix', 'vHand_fix']
    image_list = [[] for _ in name_list]
    T = len(dataloader)
    for i, (indices, model_input, ground_truth) in enumerate(tqdm(dataloader)):
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        image_list[0].append(gt)

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        jHand.textures = mesh_utils.pad_texture(jHand, 'blue')

        K_ndc = mesh_utils.intr_from_screen_to_ndc(intrinsics, H, W)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[1].append(hoi)

        # rotate by 90 degree in world frame 
        jTcp = mesh_utils.get_wTcp_in_camera_view(np.pi/2, wTc=jTc)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTcp, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[2].append(hoi)

        if i == T//2:
            # coord = plot_utils.create_coord(device, size=1)
            jHoi = mesh_utils.join_scene([jHand, jObj])
            image_list[3] = mesh_utils.render_geom_rot(jHoi, scale_geom=True, out_size=H) 
            image_list[4] = mesh_utils.render_geom_rot(jObj, scale_geom=True, out_size=H) 
            
            # rotation around z axis
            vTj = torch.FloatTensor(
                [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
            vHoi = mesh_utils.apply_transform(jHoi, vTj)
            vObj = mesh_utils.apply_transform(jObj, vTj)
            image_list[5] = mesh_utils.render_geom_rot(vHoi, scale_geom=True, out_size=H) 
            image_list[6] = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H) 
        
        jHoi = mesh_utils.join_scene([jHand, jObj])                
        vTj = torch.FloatTensor(
            [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
        vObj = mesh_utils.apply_transform(jObj, vTj)
        iObj_list = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H, bin_size=32) 
        image_list[7].append(iObj_list[i%len(iObj_list)])
        
        # HOI from fixed view point 
        scale = mesh_utils.Scale(0.5, ).cuda()
        trans = mesh_utils.Translate(0, 0.4, 0, ).cuda()
        fTj = scale.compose(trans)
        fHand = mesh_utils.apply_transform(jHand, fTj)
        fObj = mesh_utils.apply_transform(jObj, fTj)
        iHoi, iObj = mesh_utils.render_hoi_obj(fHand, fObj, 0, scale_geom=False, scale=1, bin_size=32)
        image_list[8].append(iHoi)
        
        iHand, iObj = mesh_utils.render_hoi_obj(fHand, None, 0, scale_geom=False, scale=1, bin_size=32)
        image_list[9].append(iHand)
    # save 
    for n, im_list in zip(name_list, image_list):
        for t, im in enumerate(im_list):
            image_utils.save_images(im, osp.join(save_dir, n, f'{t:03d}'))
        # save video
        im_list = image_utils.save_gif(im_list, None)
        imageio.mimsave(osp.join(save_dir, f'{n}.gif'), [im[..., :3] for im in im_list])
    return


@torch.no_grad()
def run_fig(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6, max_t=1000, T_num=None):
    device = trainer.device
    model = trainer.model
    print('save to ', save_dir)

    mesh_file = osp.join(save_dir, name + '_obj.ply')
    if not osp.exists(mesh_file):
        try:
            mesh_util.extract_mesh(
                    model.implicit_surface, 
                    N=N,
                    filepath=mesh_file,
                    volume_size=volume_size,
                )
        except ValueError:
            jObj = plot_utils.create_coord('cpu', size=0.00001)
            mesh_utils.dump_meshes([mesh_file[:-4]], jObj)
            mesh_file = mesh_file.replace('.ply', '.obj')

    jObj = mesh_utils.load_mesh(mesh_file).cuda()
    # reconstruct HOI and render in origin, 45, 60, 90 degree
    degree_list = [0, 45, 60, 90, 180, 360-60, 360-90]
    name_list = ['gt', 'overlay_hoi', 'overlay_obj']
    for d in degree_list:
        name_list += ['%d_hoi' % d, '%d_obj' % d]  


    image_list = [[] for _ in name_list]
    T = len(dataloader)

    if T_num is None:
        T_list = [0, T//2, T-1]
    else:
        T_list = np.linspace(0, T-1, T_num).astype(np.int).tolist() 
    print('len', T, T_list)
    for t, (indices, model_input, ground_truth) in enumerate(dataloader):
        if t not in T_list:
            continue
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        gt = F.adaptive_avg_pool2d(gt, (H, W))

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        K_ndc = mesh_utils.intr_from_screen_to_ndc(intrinsics, H, W)
        hoi, obj = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc)
        # image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        image_list[0].append(gt)
        image_list[1].append(hoi)  # view 0
        image_list[2].append(obj)

        for i, az in enumerate(degree_list):
            img1, img2 = mesh_utils.render_hoi_obj(jHand, jObj, az, jTc=jTc, H=H, W=W)
            image_list[3 + 2*i].append(img1)  
            image_list[3 + 2*i+1].append(img2) 
        
        # save 
        for n, im_list in zip(name_list, image_list):
            im = im_list[-1]
            image_utils.save_images(im, osp.join(save_dir, f'{t:03d}_{name}_{n}'))


def render(renderer, jHand, jObj, jTc, intrinsics, H, W, zfar=-1):
    jHand.textures = mesh_utils.pad_texture(jHand, 'blue')
    jObj.textures = mesh_utils.pad_texture(jObj, 'white')
    jMeshes = mesh_utils.join_scene([jHand, jObj])
    if jTc is None:
        image = mesh_utils.render_geom_rot(jMeshes, scale_geom=True, time_len=1, out_size=H)[0]
        mask = torch.ones_like(image)
    else:
        iMesh = renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True), 
            intrinsics, 
            jMeshes, 
            H, W, zfar,
            uv_mode=False,
            )
        image = iMesh['image']
        mask = iMesh['mask']
    return image.cpu(), mask.cpu()


def main_function(args, load_pt):
    H = W = args.reso
    device = 'cuda:0'
    config = OmegaConf.load(osp.join(load_pt.split('/ckpts')[0], 'config.yaml'))

    # load data    
    # overwrite environment
    config.environment = args.environment
    dataset, _ = get_data(config, return_val=True, val_downscale=1)
    dataloader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,)

    # build and load model 
    posenet, focal_net = get_camera(config, datasize=len(dataset)+1, H=dataset.H, W=dataset.W)

    model, trainer, render_kwargs_train, render_kwargs_test, = get_model(
        config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm, 
        device=[0], test_mode=True)
    trainer.train_dataloader = dataloader
    render_kwargs_test['H'] = H
    render_kwargs_test['W'] = W
    
    if load_pt is not None:
        state_dict = torch.load(load_pt)
        new_dict = state_dict['model']

        model_utils.load_my_state_dict(model, new_dict)
        model_utils.load_my_state_dict(posenet, state_dict['posenet'])
        model_utils.load_my_state_dict(focal_net, state_dict['focalnet'])
        
        trainer.init_camera(posenet, focal_net)
        trainer.to(device)
        trainer.eval()
        
        it = state_dict['global_step']
        name = 'it%08d' % it
        save_dir = load_pt.split('/ckpts')[0] + '/vis_clip/'
    
    os.makedirs(save_dir, exist_ok=True)

    if args.surface:
        with torch.no_grad():
            run(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.fig:
        with torch.no_grad():
            run_fig(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N, T_num=args.T_num)
    if args.video:
        with torch.no_grad():
            save_dir = load_pt.split('/ckpts')[0]
            run_video(dataloader, trainer, save_dir, 'vis_video', H=H, W=W, volume_size=args.volume_size, N=args.N)


@hydra.main(config_path='../configs', config_name='eval', version_base=None)
@slurm_utils.slurm_engine()
def main_batch(args):
    print(osp.join(args.load_dir + '*', 'ckpts/latest.pt'))
    model_list = glob(osp.join(args.load_dir + '*', 'ckpts/latest.pt'))
    for e in model_list:
        load_pt = e
        print(load_pt)
        main_function(args, load_pt)


if __name__ == '__main__':
    main_batch()    