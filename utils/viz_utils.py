import torch
import torch.nn as nn
import os.path as osp
import wandb
from jutils import image_utils, mesh_utils, plot_utils, geom_utils, hand_utils


class Visualizer(nn.Module):
    def __init__(self, cfg, log_dir) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable_bimanual = cfg.get("enable_bimanual", False)

        self.log_dir = log_dir
        self.hand_wrapper = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean)
        if self.enable_bimanual:
            self.hand_wrapper_left = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean, side="left")

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

    def render_hand(self, hA, name, log, step=None, side="right"):
        assert side in ["left", "right"]
        if side=="right": hHand, _ = self.hand_wrapper(None, hA)
        else: hHand, _ = self.hand_wrapper_left(None, hA)

        hHand.textures = mesh_utils.pad_texture(hHand, "blue")
        image_list = mesh_utils.render_geom_rot_v2(hHand)
        self.add_gif(image_list, name, log, step)

    def render_hoi(self, obj, hA, name, log, step=None, hA_left=None, nTh_left=None):
        hHand, _ = self.hand_wrapper(None, hA)
        if self.enable_bimanual:
            hHand_left, _ = self.hand_wrapper_left(None, hA_left)
            jHand_left = mesh_utils.apply_transform(
                hHand_left,
                nTh_left[:, 0],
            )
            jHand_left.textures = mesh_utils.pad_texture(jHand_left, "blue")

        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
        jHand = mesh_utils.apply_transform(
            hHand,
            nTh,
        )
        jHand.textures = mesh_utils.pad_texture(jHand, "blue")

        obj.textures = mesh_utils.pad_texture(obj, "yellow")
        if self.enable_bimanual:
            hoi = mesh_utils.join_scene([obj, jHand, jHand_left])
        else:
            hoi = mesh_utils.join_scene([obj, jHand])

        image_list = mesh_utils.render_geom_rot_v2(hoi)

        self.add_gif(image_list, name, log, step)

    def render_hA_traj(self, hA_list, name, log, step=None, device="cpu", side="right", nTh_left=None):
        assert side in ["left", "right"]
        image_list = []
        for hA in hA_list:
            hA = hA.to(device)
            if side=="right":
                hHand, _ = self.hand_wrapper(None, hA)
                nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
                jHand = mesh_utils.apply_transform(
                    hHand,
                    nTh,
                )
            else:
                assert nTh_left is not None
                hHand, _ = self.hand_wrapper_left(None, hA)
                jHand = mesh_utils.apply_transform(hHand, nTh_left[:,0])

            jHand.textures = mesh_utils.pad_texture(jHand, "blue")
            gif = mesh_utils.render_geom_rot_v2(jHand, time_len=1)
            image_list.append(gif[0])
        self.add_gif(image_list, name, log, step)
