import torch
import torch.nn as nn
import os.path as osp
import wandb
from jutils import image_utils, mesh_utils, plot_utils, geom_utils, hand_utils


class Visualizer(nn.Module):
    def __init__(self, cfg, log_dir) -> None:
        super().__init__()
        self.cfg = cfg
        self.log_dir = log_dir
        self.hand_wrapper = hand_utils.ManopthWrapper(cfg.environment.mano_dir)

    def add_image(self, image, name, log, step=None):
        fname = osp.join(self.log_dir, f"{name}_{step}")
        image_utils.save_images(image, fname)
        log[f"{name}"] = wandb.Image(fname + ".png")
        return log

    def add_gif(self, image_list, name, log, step=None):
        fname = osp.join(self.log_dir, f"{name}_{step}")
        image_utils.save_gif(image_list, fname)
        log[f"{name}"] = wandb.Video(fname + ".gif")
        return log

    def render_grids(self, grids, nTw, name, log, step=None):
        lim = self.cfg.side_lim
        image_list = mesh_utils.render_sdf_grid_rot(
            grids, half_size=lim, nTw=nTw, beta=10
        )

        self.add_gif(image_list, name, log, step)

    def render_meshes(self, meshes, name, log, step=None):
        if meshes.isempty():
            print("emptey meshes! ")
            meshes = plot_utils.create_coord(meshes.device, len(meshes))
        image_list = mesh_utils.render_geom_rot_v2(meshes)

        self.add_gif(image_list, name, log, step)

    def render_hand(self, hA, name, log, step=None):
        hHand, _ = self.hand_wrapper(None, hA)
        hHand.textures = mesh_utils.pad_texture(hHand, "blue")
        image_list = mesh_utils.render_geom_rot_v2(hHand)
        self.add_gif(image_list, name, log, step)

    def render_hoi(self, obj, hA, name, log, step=None):
        hHand, _ = self.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
        jHand = mesh_utils.apply_transform(
            hHand,
            nTh,
        )
        jHand.textures = mesh_utils.pad_texture(jHand, "blue")

        obj.textures = mesh_utils.pad_texture(obj, "yellow")
        hoi = mesh_utils.join_scene([obj, jHand])
        image_list = mesh_utils.render_geom_rot_v2(hoi)

        self.add_gif(image_list, name, log, step)

    def render_hA_traj(self, hA_list, name, log, step=None, device="cpu"):
        image_list = []
        for hA in hA_list:
            hA = hA.to(device)
            hHand, _ = self.hand_wrapper(None, hA)
            nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
            jHand = mesh_utils.apply_transform(
                hHand,
                nTh,
            )
            jHand.textures = mesh_utils.pad_texture(jHand, "blue")

            gif = mesh_utils.render_geom_rot_v2(jHand, time_len=1)
            image_list.append(gif[0])
        self.add_gif(image_list, name, log, step)
