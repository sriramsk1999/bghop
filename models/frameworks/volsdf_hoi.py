import copy
import functools
import os.path as osp
from collections import OrderedDict

import pytorch3d.transforms.rotation_conversions as rot_cvt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jutils import geom_utils, hand_utils, image_utils, mesh_utils, model_utils
from pytorch3d.loss.chamfer import knn_points
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

from models.articulation import get_artnet
from models.blending import hard_rgb_blend, volumetric_rgb_blend
from models.cameras.gt import PoseNet
from models.frameworks.volsdf import SingleRenderer, VolSDF
from models.pix_sampler import PixelSampler, get_pixel_sampler
from models.sd import SDLoss
from utils import rend_util
from utils.logger import Logger


class RelPoseNet(nn.Module):
    """Per-frame R,t correction @ base s,R,t
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_frames,
        learn_R,
        learn_t,
        init_pose,
        learn_base_s,
        learn_base_R,
        learn_base_t,
        **kwargs,
    ):
        """
        Args:
            num_frames (_type_): N
            learn_R (_type_): True/False
            learn_T (_type_): True/False
            init_pose (_type_): base pose srt initialziation (1, 4, 4)
            learn_base_s (_type_): True/False
            learn_base_R (_type_): True/False
            learn_base_t (_type_): True/False
        """
        super().__init__()
        if init_pose is None:
            init_pose = torch.eye(4).unsqueeze(0)
        base_r, base_t, base_s = geom_utils.homo_to_rt(init_pose)
        # use axisang
        self.base_r = nn.Parameter(
            geom_utils.matrix_to_axis_angle(base_r), learn_base_R
        )
        self.base_s = nn.Parameter(base_s[..., 0:1], learn_base_s)
        self.base_t = nn.Parameter(base_t, learn_base_t)

        # ??
        self.r = nn.Parameter(
            torch.zeros(size=(num_frames, 3), dtype=torch.float32),
            requires_grad=learn_R,
        )  # (N, 3)
        self.t = nn.Parameter(
            torch.zeros(size=(num_frames, 3), dtype=torch.float32),
            requires_grad=learn_t,
        )  # (N, 3)

    def forward(self, cam_id, *args, **kwargs):
        r = torch.gather(self.r, 0, torch.stack(3 * [cam_id], -1))  # (3, ) axis-angle
        t = torch.gather(self.t, 0, torch.stack(3 * [cam_id], -1))  # (3, )
        frameTbase = geom_utils.axis_angle_t_to_matrix(r, t)

        N = len(r)
        base = geom_utils.rt_to_homo(
            rot_cvt.axis_angle_to_matrix(self.base_r),
            self.base_t,
            self.base_s.repeat(1, 3),
        )
        base = base.repeat(N, 1, 1)

        frame_pose = frameTbase @ base
        return frame_pose


class VolSDFHoi(nn.Module):
    # TODO: this can be disentangled better with renderer...
    # TODO: should handle hybrid rerpesentation.
    def __init__(
        self,
        data_size=100,
        beta_init=0.1,
        speed_factor=1.0,
        input_ch=3,
        W_geo_feat=-1,
        obj_bounding_radius=3.0,
        use_nerfplusplus=False,
        use_tinycuda=False,
        sdf_rep="net",
        surface_cfg=dict(),
        radiance_cfg=dict(),
        oTh_cfg=dict(),
        hA_cfg=dict(),
    ):
        super().__init__()
        self.volsdf = VolSDF(
            beta_init,
            speed_factor,
            input_ch,
            W_geo_feat,
            obj_bounding_radius,
            use_nerfplusplus,
            use_tinycuda,
            surface_cfg,
            radiance_cfg,
            sdf_rep=sdf_rep,
        )

        if oTh_cfg["mode"] == "learn":
            self.oTh = RelPoseNet(data_size, init_pose=None, **oTh_cfg)
        elif oTh_cfg["mode"] == "gt":
            self.oTh = PoseNet("hTo", True)
        else:
            raise NotImplementedError("Not implemented oTh.model: %s" % oTh_cfg["mode"])
        # initialize uv texture of hand
        t_size = 32
        uv_text = torch.ones([1, t_size, t_size, 3])
        uv_text[..., 2] = 0
        self.uv_text = nn.Parameter(uv_text)
        # TODO
        self.hand_shape = nn.Parameter(torch.zeros(1, 10))
        self.uv_text_init = False

        # hand articulation
        hA_mode = hA_cfg.pop("mode")
        self.hA_net = get_artnet(hA_mode, hA_cfg)

    @property
    def implicit_surface(self):
        return self.volsdf.implicit_surface

    def forward_ab(self):
        return self.volsdf.forward_ab()

    def forward_surface(self, x: torch.Tensor):
        return self.volsdf.forward_surface(x)

    def forward_surface_with_nablas(self, x: torch.Tensor):
        return self.volsdf.forward_surface_with_nablas(x)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        return self.volsdf.forward(x, view_dirs)


class MeshRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    def get_cameras(self, cTw, pix_intr, H, W):
        K = mesh_utils.intr_from_screen_to_ndc(pix_intr, H, W)
        f, p = mesh_utils.get_fxfy_pxpy(K)
        if cTw is None:
            cameras = PerspectiveCameras(f, p, device=pix_intr.device)
        else:
            rot, t, _ = geom_utils.homo_to_rt(cTw)
            # cMesh
            cameras = PerspectiveCameras(
                f, p, R=rot.transpose(-1, -2), T=t, device=cTw.device
            )
        return cameras

    def forward(self, cTw, pix_intr, meshes: Meshes, H, W, far, **kwargs):
        cameras = self.get_cameras(None, pix_intr, H, W)

        # apply cameraTworld outside of rendering to support scaling.
        cMeshes = mesh_utils.apply_transform(meshes, cTw)  # to allow scaling
        if kwargs.get("soft", True):
            render_fn = mesh_utils.render_soft
        else:
            render_fn = mesh_utils.render_mesh
        image = render_fn(
            cMeshes,
            cameras,
            rgb_mode=True,
            depth_mode=True,
            normal_mode=True,
            xy_mode=True,
            uv_mode=kwargs.get("uv_mode", True),
            out_size=max(H, W),
        )
        # apply cameraTworld for depth
        # depth is in camera-view but is in another metric!
        # convert the depth unit to the normalized unit.
        with torch.no_grad():
            _, _, cTw_scale = geom_utils.homo_to_rt(cTw)
            wTc_scale = 1 / cTw_scale.mean(-1)
        image["depth"] = wTc_scale * image["depth"]
        image["depth"] = image["mask"] * image["depth"] + (1 - image["mask"]) * far
        return image


class Trainer(nn.Module):
    @staticmethod
    def lin2img(x):
        H = int(x.shape[1] ** 0.5)
        return rearrange(x, "n (h w) c -> n c h w", h=H, w=H)

    def __init__(self, model: VolSDFHoi, device_ids=[0], batched=True, args=None, test_mode=False):
        super().__init__()
        self.obj_radius = args.model.obj_bounding_radius
        self.args = args
        self.H: int = 224  # training reso
        self.W: int = 224  # training reso

        self.train_dataloader = None
        self.val_dataloader = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler = None
        self.pixel_sampler: PixelSampler = get_pixel_sampler(
            args.pixel_sampler.name, args.pixel_sampler, args
        )

        self.model = model
        self.renderer = SingleRenderer(model)
        self.mesh_renderer = MeshRenderer()
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(
                self.renderer, device_ids=device_ids, dim=1 if batched else 0
            )
        self.device = device_ids[0]

        self.posenet: nn.Module = None
        self.focalnet: nn.Module = None
        self.hand_wrapper = hand_utils.ManopthWrapper(args.environment.mano_dir)

        if not test_mode:
            self.sd_loss = SDLoss(
                args.novel_view.diffuse_ckpt, args, **args.novel_view.sd_para
            )
            self.sd_loss.init_model(self.device)

    def init_camera(self, posenet, focalnet):
        self.posenet = posenet
        self.focalnet = focalnet

    def get_jHand_camera(self, indices, model_input, ground_truth, H, W):
        jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)

        hA = self.model.hA_net(indices, model_input, None)
        # hand FK
        N = len(hA)
        hand_shape = self.model.hand_shape.repeat(N, 1)
        uv_text = self.model.uv_text.repeat(N, 1, 1, 1)
        hHand, _ = self.hand_wrapper(None, hA, texture=uv_text, th_betas=hand_shape)
        jHand = mesh_utils.apply_transform(hHand, jTh)

        return jHand, jTc, jTh, intrinsics

    def render(
        self,
        jHand,
        jTc,
        intrinsics,
        render_kwargs,
        use_surface_render="sphere_tracing",
        blend="hard",
    ):
        H, W = render_kwargs["H"], render_kwargs["W"]

        norm = mesh_utils.get_camera_dist(wTc=jTc)
        render_kwargs["far"] = zfar = (norm + self.obj_radius).cpu().item()
        render_kwargs["near"] = znear = (norm - self.obj_radius).cpu().item()

        model = self.model
        if use_surface_render:
            assert (
                use_surface_render == "sphere_tracing"
                or use_surface_render == "root_finding"
            )
            from models.ray_casting import surface_render

            render_fn = functools.partial(
                surface_render,
                model=model,
                ray_casting_algo=use_surface_render,
                ray_casting_cfgs={
                    "near": render_kwargs["near"],
                    "far": render_kwargs["far"],
                },
            )
        else:
            render_fn = self.renderer

        # mesh rendering
        iHand = self.mesh_renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True),
            intrinsics,
            jHand,
            **render_kwargs,
        )

        # volumetric rendering
        # rays in canonical object frame
        rays_o, rays_d, select_inds = rend_util.get_rays(
            jTc, intrinsics, H, W, N_rays=-1
        )
        rgb_obj, depth_v, extras = render_fn(
            rays_o, rays_d, detailed_output=True, **render_kwargs
        )

        iObj = {
            "rgb": rgb_obj,
            "depth": depth_v,
        }
        if use_surface_render:
            iObj["mask"] = extras["mask_surface"]
        else:
            iObj["mask"] = extras["mask_volume"]

        iHoi = self.blend(iHand, iObj, select_inds, zfar, znear, method="hard")

        # matting
        pred_hand_select = (iHand["depth"] < iObj["depth"]).float().unsqueeze(-1)

        if not self.training:
            rgb = iHoi["rgb"].reshape(1, H, W, 3).permute([0, 3, 1, 2])
            pred_hand_select = pred_hand_select.reshape(1, 1, H, W)
        rtn = {}
        rtn["image"] = rgb.cpu()
        rtn["hand"] = iHand["image"].cpu()
        rtn["obj"] = iObj["rgb"].reshape(1, H, W, 3).permute([0, 3, 1, 2]).cpu()
        rtn["label"] = iHoi["label"].reshape(1, H, W, 3).permute([0, 3, 1, 2])
        return rtn

    def get_jTc(self, indices, model_input, ground_truth):
        device = self.device
        # palmTcam
        wTc = self.posenet(
            indices,
            model_input,
            ground_truth,
        )
        wTc_n = self.posenet(
            model_input["inds_n"].to(device),
            model_input,
            ground_truth,
        )

        # identity
        hTw = geom_utils.inverse_rt(mat=model_input["wTh"].to(device), return_mat=True)
        hTw_n = geom_utils.inverse_rt(
            mat=model_input["wTh_n"].to(device), return_mat=True
        )

        # TODO: change here!
        oTh = self.model.oTh(indices.to(device), model_input, ground_truth)
        oTh_n = self.model.oTh(
            model_input["inds_n"].to(device), model_input, ground_truth
        )

        # NOTE: the mesh / vol rendering needs to be in the same coord system (joint), in order to compare their depth!!
        onTo = model_input["onTo"].to(device)  # scale: 10
        onTo_n = model_input["onTo_n"].to(device)

        oTc = oTh @ hTw @ wTc
        oTc_n = oTh_n @ hTw_n @ wTc_n

        jTh = onTo @ oTh
        jTh_n = onTo_n @ oTh_n
        jTc = onTo @ oTc  # wTc
        jTc_n = onTo @ oTc_n  # wTc_n

        return jTc, jTc_n, jTh, jTh_n

    def blend(
        self,
        iHand,
        iObj,
        select_inds,
        znear,
        zfar,
        method="hard",
        sigma=1e-4,
        gamma=1e-4,
        background_color=(1.0, 1.0, 1.0),
        **kwargs,
    ):
        N = len(iHand["image"])
        # change and select inds
        # iHand['rgb'] = iHand['image'].view(N, 3, -1).transpose(-1, -2)
        iHand["rgb"] = rearrange(iHand["image"], "n c h w -> n (h w) c")
        iHand["normal"] = rearrange(iHand["normal"], "n c h w -> n (h w) c")
        iHand["depth"] = iHand["depth"].view(N, -1)
        iHand["mask"] = iHand["mask"].view(N, -1)

        iHand["rgb"] = torch.gather(iHand["rgb"], 1, torch.stack(3 * [select_inds], -1))
        iHand["normal"] = torch.gather(
            iHand["normal"], 1, torch.stack(3 * [select_inds], -1)
        )
        iHand["depth"] = torch.gather(iHand["depth"], 1, select_inds).float()
        iHand["mask"] = torch.gather(iHand["mask"], 1, select_inds).float()

        blend_params = BlendParams(sigma, gamma, background_color)
        blend_label = BlendParams(sigma, gamma, (0.0, 0.0, 0.0))
        iHoi = {}
        if method == 'vol':
            hand_label = F.one_hot(torch.zeros_like(iHand['mask']).long(), num_classes=3).float()
            obj_label = F.one_hot(torch.ones_like(iObj['mask']).long(), num_classes=3).float()
            iHoi['label'] = volumetric_rgb_blend(
                (hand_label, obj_label),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_label, znear, zfar)[..., 0:3]
            iHoi['depth'] = torch.minimum(iHand['depth'], iObj['depth'])
            # TODO: check pytorch3d normal!!!???
            iHoi['normal'] = volumetric_rgb_blend(
                (iHand['normal'], iObj['normal_c']),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_label, znear, zfar)[..., 0:3]  

            rgba = volumetric_rgb_blend(
                (iHand['rgb'], iObj['rgb']),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_params, znear, zfar)
            iHoi['rgb'], iHoi['mask'] = rgba.split([3, 1], -1)
            iHoi['mask'] = iHoi['mask'].squeeze(-1)
        elif method == "hard":
            rgba = hard_rgb_blend(
                (iHand["rgb"], iObj["rgb"]),
                (iHand["depth"], iObj["depth"]),
                (iHand["mask"], iObj["mask"]),
                blend_params,
            )
            iHoi["rgb"], iHoi["mask"] = rgba.split([3, 1], -1)
            iHoi["mask"] = iHoi["mask"].squeeze(-1)

            hand_label = torch.zeros_like(iHand["mask"]).long()
            obj_label = torch.ones_like(iHand["mask"]).long()
            iHoi["label"] = hard_rgb_blend(
                (
                    F.one_hot(hand_label, num_classes=3),
                    F.one_hot(obj_label, num_classes=3),
                ),
                (iHand["depth"], iObj["depth"]),
                (iHand["mask"], iObj["mask"]),
                blend_label,
            )[..., 0:3]
        else:
            raise NotImplementedError("blend method %s" % method)
        return iHoi

    def get_reproj_loss(
        self, losses, extras, ground_truth, model_input, render_kwargs_train
    ):
        """losses that compare with input -- cannot be applied to novel-view

        :param iHoi: _description_
        :param ground_truth: _description_
        :param model_input: _description_
        :param select_inds: _description_
        :param losses: _description_
        :param render_kwargs_train: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        args = self.args
        device = self.device
        iHoi = extras["iHoi"]
        label_target = extras["label_target"]

        if args.training.w_hand_mask > 0:
            obj_mask = model_input["obj_mask"].float()
            N = obj_mask.shape[0]
            losses["loss_hand_mask"] = args.training.w_hand_mask * F.l1_loss(
                obj_mask * extras["iHand_full"]["mask"].view(N, -1),
                obj_mask * model_input["hand_mask"],
            )

        losses["loss_mask"] = args.training.w_mask * F.l1_loss(
            iHoi["label"], label_target
        )

        if args.training.w_contour > 0:
            iHand = extras["iHand"]
            gt_cont = model_input["hand_contour"].to(device)
            x_nn = knn_points(gt_cont, iHand["xy"][..., :2], K=1)
            cham_x = x_nn.dists.mean()
            losses["loss_contour"] = args.training.w_contour * cham_x
        return losses

    def get_reg_loss(self, losses, extras):
        """can be applied to original and novel view"""
        args = self.args
        if args.training.w_eikonal > 0:
            # always compute this even when the weight is zero --> populate loss to a correct shape
            nablas_norm = extras["implicit_nablas_norm"]
            losses["loss_eikonal"] = args.training.w_eikonal * F.mse_loss(
                nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction="mean"
            )

    def get_diffusion_volume(
        self,
        extras,
        lim=None,
        tsdf=None,
        reso=None,
        offset=None,
        up_factor=1,
        soft="hard",
    ):
        """
        Args:
            extras (_type_): _description_
            lim (_type_, optional): _description_. Defaults to None.
            tsdf (_type_, optional): _description_. Defaults to None.
            reso (_type_, optional): _description_. Defaults to None.
            offset float or tensor in shape of (N, 3): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        # get grid of TSDF
        lim = self.sd_loss.model.cfg.side_lim if lim is None else lim
        tsdf = self.sd_loss.model.cfg.tsdf if tsdf is None else tsdf
        reso = self.sd_loss.model.cfg.side_x if reso is None else reso
        hA = extras["hA"]
        B = len(hA)
        device = hA.device

        nXyz = mesh_utils.create_sdf_grid(
            B, int(up_factor * reso), lim, "zyx", device=device
        )  # (B, H, H, H, 3) in range of (-1, 1)
        nXyz = nXyz.reshape(B, -1, 3)
        if offset is None:
            offset = torch.zeros([B, 3], device=device)
        elif isinstance(offset, float):
            offset = torch.ones([B, 3], device=device) * offset

        nXyz = nXyz + offset.reshape(B, 1, 3)
        hTn = hand_utils.get_nTh(hA=hA, hand_wrapper=self.hand_wrapper, inverse=True)
        jTh = extras["jTh"]
        jTn = jTh @ hTn.detach()
        _, _, jTn_scale = geom_utils.homo_to_rt(jTn)  # (B, 3)
        jXyz = mesh_utils.apply_transform(nXyz, jTn)
        # query model to get sdf
        jSdf, nblas, h = self.model.implicit_surface.forward_with_nablas(
            jXyz
        )  # (N, R, ?)
        nSdf = jSdf / jTn_scale[..., 0:1]
        up_reso = int(up_factor * reso)
        nSdf = nSdf.reshape(B, 1, up_reso, up_reso, up_reso)  # fuzzy!!!
        nXyz = nXyz.reshape(B, up_reso, up_reso, up_reso, 3)
        if up_factor > 1:
            nSdf = F.adaptive_avg_pool3d(nSdf, reso)

            nXyz = nXyz.permute([0, 4, 1, 2, 3])
            nXyz = F.adaptive_avg_pool3d(nXyz, reso)
            nXyz = nXyz.permute([0, 2, 3, 4, 1])

        rtn = self.sd_loss.model.set_inputs(
            {"nSdf": nSdf, "hA": hA, "nXyz": nXyz}, 0, soft=soft
        )
        rtn["hA"] = hA
        rtn["offset"] = offset
        rtn["raw"] = nSdf

        return rtn

    def get_fullvol_reg_loss(self, losses, extras, model_input, it, **kwargs):
        """apply to full volume query"""
        args = self.args
        if args.training.w_diffuse > 0:
            r = args.get("diff_r", 0.02)
            up_factor = args.get("diff_up", 1)
            offset = torch.rand([1, 3], device=extras["hA"].device) * r * 2 - r
            img = self.get_diffusion_volume(
                extras, offset=offset, up_factor=up_factor, **kwargs
            )
            grad, rtn = self.sd_loss.apply_sd(
                img,
                text=model_input["text"],
                weight=args.training.w_diffuse,
                **args.novel_view.loss,
                it=it,
            )
            losses.update(rtn["losses"])
            extras["diffusion_inp"] = img
            extras["diff_t"] = rtn["t"]
            extras["diff_w"] = rtn["w"]
            if "start_pred" in rtn:
                extras["diffusion_gt"] = rtn["start_pred"]
        return losses

    def get_temporal_loss(self, losses, extras):
        """only makes sense when consective frame make sense~"""
        args = self.args
        # temporal smoothness loss
        # jJoints, hJoints?
        if args.training.w_t_hand > 0:
            jJonits_diff = ((extras["jJoints"] - extras["jJoints_n"]) ** 2).mean()
            w = args.training.get("w_t_hand_j", args.training.w_t_hand)
            losses["loss_dt_joint_j"] = 0.5 * w * (jJonits_diff)

            # does not account for translation.
            centered_joints = extras["cJoints"] - extras["cJoints"][:, 5:6].detach()
            centered_joints_n = (
                extras["cJoints_n"] - extras["cJoints_n"][:, 5:6].detach()
            )
            cJoints_diff = ((centered_joints - centered_joints_n) ** 2).mean()
            w = args.training.get("w_t_hand_c", args.training.w_t_hand)
            losses["loss_dt_joint_c"] = 0.5 * w * (cJoints_diff)

            w = args.training.get("w_t_hand_i", 0)
            iJoints_diff = ((extras["iJoints"] - extras["iJoints_n"]) ** 2).mean()
            losses["loss_dt_joint_i"] = w * iJoints_diff

            # cJonits_diff = ((extras['cJoints'] - extras['cJoints_n'])**2).mean()
        return losses

    def proj3d(self, cPoints, intrinsics):
        """
        Args:
            cPoints (_type_): (N, P, 3)
            intrinsics (_type_): (N, 3, 4)

        Returns:
            _type_: _description_
        """
        cPoints = torch.cat(
            [cPoints, torch.ones_like(cPoints[..., :1])], dim=-1
        )  # (N, P, 4)
        iPoints = cPoints @ intrinsics.transpose(-1, -2)  # (N, P, 3)
        iPoints = iPoints[..., :2] / iPoints[..., 2:3]
        return iPoints

    def forward(
        self,
        args,
        indices,
        model_input,
        ground_truth,
        render_kwargs_train: dict,
        it: int,
        calc_loss=True,
    ):
        device = self.device
        N = len(indices)
        indices = indices.to(device)
        model_input = model_utils.to_cuda(model_input, device)
        query_full_vol = (
            self.training
            and self.args.training.get("query_full_vol", False)
            and it % 2 == 0
            and it >= self.args.training.warmup
        )

        jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)

        # NOTE: znear and zfar is important: distance of camera center to world origin
        cam_norm = jTc[..., 0:4, 3]
        cam_norm = cam_norm[..., 0:3] / cam_norm[..., 3:4]  # (N, 3)
        norm = torch.norm(cam_norm, dim=-1)

        render_kwargs_train["far"] = zfar = (
            (norm + args.model.obj_bounding_radius).cpu().item()
        )
        render_kwargs_train["near"] = znear = (
            (norm - args.model.obj_bounding_radius).cpu().item()
        )

        H = render_kwargs_train["H"]
        W = render_kwargs_train["W"]
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)
        intrinsics_n = self.focalnet(
            model_input["inds_n"].to(device), model_input, ground_truth, H=H, W=W
        )

        # 2. RENDER MY SCENE
        # 2.1 RENDER OBJECT
        # volumetric rendering
        # rays in canonical object frame
        if self.training:
            select_inds = self.pixel_sampler(model_input, H, W, args.data.N_rays, it)
            # select_inds = None
            rays_o, rays_d, select_inds = rend_util.get_rays(
                jTc, intrinsics, H, W, N_rays=args.data.N_rays, inds=select_inds
            )
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                jTc, intrinsics, H, W, N_rays=-1
            )
        rgb_obj, depth_v, extras = self.renderer(
            rays_o, rays_d, detailed_output=True, **render_kwargs_train
        )
        # rgb: (N, R, 3), mask/depth: (N, R, ), flow: (N, R, 2?)
        iObj = {
            "rgb": rgb_obj,
            "depth": depth_v,
            "mask": extras["mask_volume"],
        }
        if "normals_volume" in extras:
            iObj["normal"] = extras["normals_volume"]
            jObj_normal = iObj["normal"] * iObj["mask"][..., None]

            cTj = geom_utils.inverse_rt(mat=jTc, return_mat=True)
            cTj_trans = Transform3d(matrix=cTj.transpose(-1, -2), device=cTj.device)
            # this is because my weird def of pytorch3d mesh rendering of normal in mesh_utils.render
            cObj_normal = cTj_trans.transform_normals(
                jObj_normal
            )  # need to renorm! coz of scale happens in cTj...
            cObj_normal[..., 1:3] *= -1  # xyz points to left, up, outward
            cHoi_norm = torch.norm(cObj_normal, dim=-1, keepdim=True) + 1e-5
            cObj_normal = cObj_normal / cHoi_norm
            iObj["normal_c"] = cObj_normal * iObj["mask"][..., None]

        extras["iObj"] = iObj
        extras["select_inds"] = select_inds

        # 2.2 RENDER HAND
        # hand FK
        hA = self.model.hA_net(indices, model_input, None)
        hA_n = self.model.hA_net(model_input["inds_n"].to(device), model_input, None)
        hHand, hJoints = self.hand_wrapper(
            None, hA, texture=self.model.uv_text, th_betas=self.model.hand_shape
        )
        jHand = mesh_utils.apply_transform(hHand, jTh)
        jJoints = mesh_utils.apply_transform(hJoints, jTh)
        cJoints = mesh_utils.apply_transform(
            jJoints, geom_utils.inverse_rt(mat=jTc, return_mat=True)
        )
        extras["jTc"] = jTc
        extras["jTh"] = jTh
        extras["hand"] = jHand
        extras["jJoints"] = jJoints
        extras["iJoints"] = self.proj3d(cJoints, intrinsics)
        extras["cJoints"] = cJoints

        hHand_n, hJoints_n = self.hand_wrapper(
            None, hA_n, texture=self.model.uv_text, th_betas=self.model.hand_shape
        )
        jJoints_n = mesh_utils.apply_transform(hJoints_n, jTh_n)
        cJoints_n = mesh_utils.apply_transform(
            jJoints_n, geom_utils.inverse_rt(mat=jTc_n, return_mat=True)
        )
        extras["jJoints_n"] = jJoints_n
        extras["iJoints_n"] = self.proj3d(cJoints_n, intrinsics_n)
        extras["cJoints_n"] = cJoints_n

        iHand = self.mesh_renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True),
            intrinsics,
            jHand,
            **render_kwargs_train,
        )
        extras["iHand_full"] = {}
        for k, v in iHand.items():
            extras["iHand_full"][k] = v
        extras["iHand"] = iHand
        extras["hA"] = hA

        # 2.3 BLEND!!!
        # blended rgb, detph, mask, flow
        iHoi = self.blend(iHand, iObj, select_inds, znear, zfar, **args.blend_train)
        extras["iHoi"] = iHoi

        # 3. GET GROUND TRUTH SUPERVISION
        # [B, N_rays, 3]
        target_rgb = torch.gather(
            ground_truth["rgb"].to(device), 1, torch.stack(3 * [select_inds], -1)
        )
        # [B, N_rays]
        target_mask = torch.gather(
            model_input["object_mask"].to(device), 1, select_inds
        ).float()
        # [B, N_rays]
        target_obj = torch.gather(
            model_input["obj_mask"].to(device), 1, select_inds
        ).float()
        target_hand = torch.gather(
            model_input["hand_mask"].to(device), 1, select_inds
        ).float()
        extras["target_rgb"] = target_rgb
        extras["target_mask"] = target_mask
        extras["target_obj"] = target_obj
        extras["target_hand"] = target_hand

        # masks for mask loss: # (N, P, 2)
        # GT is marked as hand not object, AND predicted object depth is behind
        ignore_obj = ((target_hand > 0) & ~(target_obj > 0)) & (
            iObj["depth"] > iHand["depth"]
        )
        ignore_hand = (~(target_hand > 0) & (target_obj > 0)) & (
            iObj["depth"] < iHand["depth"]
        )
        label_target = torch.stack(
            [target_hand, target_obj, torch.zeros_like(target_obj)], -1
        )  # (N, P, 3)
        extras["label_target"] = label_target

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras["implicit_nablas"]

        # [B, N_rays, ]
        # ---------- OPTION1: just flatten and use all nablas
        # nablas = nablas.flatten(-3, -2)

        # ---------- OPTION2: using only one point each ray: this may be what the paper suggests.
        # @ VolSDF section 3.5, "combine a SINGLE random uniform space point and a SINGLE point from \mathcal{S} for each pixel"
        _, _ind = extras["visibility_weights"][..., : nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(
            nablas,
            dim=-2,
            index=_ind[..., None, None].repeat([*(len(nablas.shape) - 1) * [1], 3]),
        )

        eik_bounding_box = args.model.obj_bounding_radius
        eikonal_points = (
            torch.empty_like(nablas)
            .uniform_(-eik_bounding_box, eik_bounding_box)
            .to(device)
        )
        _, nablas_eik, _ = self.model.implicit_surface.forward_with_nablas(
            eikonal_points
        )

        # hacky way to concat two nablas_norm, (norm first before concat, instead of concat and then norm)
        nablas_norm1 = nablas.norm(2, dim=-1)
        nablas_norm2 = nablas_eik.norm(2, dim=-1)
        nablas_norm = torch.cat([nablas_norm1, nablas_norm2], dim=-1)

        # [B, N_rays, N_pts]
        # nablas_norm = torch.norm(nablas_all, dim=-1)
        extras["implicit_nablas_norm"] = nablas_norm

        if not calc_loss:
            # early return
            return extras
        losses = OrderedDict()
        self.get_reg_loss(losses, extras)
        if query_full_vol:
            self.get_fullvol_reg_loss(losses, extras, model_input, it)
        else:
            self.get_reproj_loss(
                losses, extras, ground_truth, model_input, render_kwargs_train
            )
            self.get_temporal_loss(losses, extras)

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses["total"] = loss

        alpha, beta = self.model.forward_ab()
        alpha = alpha.data
        beta = beta.data
        extras["scalars"] = {"beta": beta, "alpha": alpha}

        extras["obj_ignore"] = ignore_obj
        extras["hand_ignore"] = ignore_hand

        extras["obj_mask_target"] = model_input["obj_mask"]
        extras["hand_mask_target"] = model_input["hand_mask"]
        extras["mask_target"] = model_input["object_mask"]
        # extras['flow'] = iHoi['flow']
        extras["intrinsics"] = intrinsics

        return OrderedDict([("losses", losses), ("extras", extras)])

    def val_3d(
        self,
        logger: Logger,
        ret,
        to_img_fn,
        it,
        render_kwargs_test,
        val_ind=None,
        val_in=None,
        val_gt=None,
    ):
        offset = torch.zeros([1, 3], device=self.device)
        rtn = self.get_diffusion_volume(ret, offset=offset)  # ['image'] , ['hA]
        logger.log_metrics({f"scalars/max_step": self.sd_loss.max_step}, it)
        logger.log_metrics({f"scalars/min_value": rtn["image"].min()}, it)
        grad, sd_rtn = self.sd_loss.apply_sd(
            rtn,
            text=val_in["text"],
            weight=self.args.training.w_diffuse,
            **self.args.novel_view.loss,
            it=it,
            debug=True,
        )
        pred = self.sd_loss.model.decode_samples(rtn, offset)
        jObj = pred["jObj"]
        hA = rtn["hA"]
        hHand, _ = self.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
        jHand = mesh_utils.apply_transform(hHand, nTh)
        jHand.textures = mesh_utils.pad_texture(jHand, "blue")
        jHoi = mesh_utils.join_scene([jHand, jObj])
        image_list = mesh_utils.render_geom_rot(jHoi, scale_geom=True)
        logger.add_gifs(image_list, "vol/jHoi", it)

        voxel = rtn["image"]
        jObj.textures = mesh_utils.pad_texture(
            jObj,
        )
        nTw = mesh_utils.get_nTw(jObj)
        image_list = mesh_utils.render_geom_rot_v2(jObj, nTw=nTw)
        logger.add_gifs(image_list, "vol/jObj_mesh", it)
        lim = self.sd_loss.model.cfg.side_lim
        image_list = mesh_utils.render_sdf_grid_rot(voxel, nTw=nTw, half_size=lim)
        logger.add_gifs(image_list, "vol/jObj_voxel", it)

        jObj_gt = self.sd_loss.model.decode_samples(
            {"image": sd_rtn["start_pred"]}, offset
        )["jObj"]
        if "hand_pred" in sd_rtn:
            hA, _, _, _ = self.sd_loss.model.hand_cond.grid2pose_sgd(sd_rtn["hand_pred"])
            jHand, _ = self.hand_wrapper(nTh, hA)
            jHand.textures = mesh_utils.pad_texture(jHand, "blue")
        jHoi_gt = mesh_utils.join_scene([jHand, jObj_gt])
        image_list = mesh_utils.render_geom_rot(jHoi_gt, scale_geom=True)
        logger.add_gifs(image_list, "vol/jHoi_gt", it)

        hh = ww = int(val_gt["rgb"].size(1) ** 0.5)
        H, W = render_kwargs_test["H"], render_kwargs_test["W"]
        jTh = ret["jTh"]
        jTc = ret["jTc"]
        cTj = geom_utils.inverse_rt(mat=jTc, return_mat=True)
        nHoi_gt = jHoi_gt
        cTn = cTj @ jTh @ geom_utils.inverse_rt(mat=nTh, return_mat=True)
        cHoi_gt = mesh_utils.apply_transform(nHoi_gt, cTn)
        intrinsics = ret["intrinsics"]
        cameras = self.mesh_renderer.get_cameras(None, intrinsics, H, W)
        # K_ndc = mesh_utils.intr_from_screen_to_ndc(ret['intrinsics'], H, W)
        iHoi_gt = mesh_utils.render_mesh(cHoi_gt, cameras, out_size=H)

        gt = val_gt["rgb"].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        inp = F.adaptive_avg_pool2d(gt, (H, W))
        out = image_utils.blend_images(iHoi_gt["image"], inp, iHoi_gt["mask"])
        logger.add_imgs(out, "vol/cHoi_gt", it)

    def val(
        self,
        logger: Logger,
        ret,
        to_img_fn,
        it,
        render_kwargs_test,
        val_ind=None,
        val_in=None,
        val_gt=None,
    ):
        mesh_utils.dump_meshes(
            osp.join(logger.log_dir, "hand_meshes/%08d" % it), ret["hand"]
        )
        logger.add_meshes("hand", osp.join("hand_meshes/%08d_00.obj" % it), it)

        # 3D
        if self.args.novel_view.mode == "3d":
            self.val_3d(
                logger, ret, to_img_fn, it, render_kwargs_test, val_ind, val_in, val_gt
            )

        # vis reproje
        mask = torch.cat(
            [
                to_img_fn(ret["hand_mask_target"].unsqueeze(-1).float())
                .repeat(1, 3, 1, 1)
                .cpu(),
                to_img_fn(ret["obj_mask_target"].unsqueeze(-1))
                .repeat(1, 3, 1, 1)
                .cpu(),
                to_img_fn(ret["mask_target"].unsqueeze(-1)).repeat(1, 3, 1, 1).cpu(),
                to_img_fn(ret["label_target"].cpu()),
            ],
            -1,
        )

        logger.add_imgs(mask, "gt/hoi_mask_gt", it)

        mask = torch.cat(
            [
                to_img_fn(ret["iHand"]["mask"].unsqueeze(-1)).repeat(1, 3, 1, 1).cpu(),
                to_img_fn(ret["iObj"]["mask"].unsqueeze(-1)).repeat(1, 3, 1, 1).cpu(),
                to_img_fn(ret["iHoi"]["mask"].unsqueeze(-1)).repeat(1, 3, 1, 1).cpu(),
                to_img_fn(ret["iHoi"]["label"]).cpu(),
            ],
            -1,
        )
        logger.add_imgs(mask, "hoi/hoi_mask_pred", it)


def get_model(args, data_size=-1, **kwargs):
    if args.model.get("sdf_rep", "net") != "net":
        surface_cfg = args.model.surface
        radiance_cfg = args.model.radiance
    else:
        surface_cfg = {
            "use_siren": args.model.surface.get(
                "use_siren", args.model.get("use_siren", False)
            ),
            "embed_multires": args.model.surface.get("embed_multires", 6),
            "radius_init": args.model.surface.get("radius_init", 1.0),
            "geometric_init": args.model.surface.get("geometric_init", True),
            "D": args.model.surface.get("D", 8),
            "W": args.model.surface.get("W", 256),
            "skips": args.model.surface.get("skips", [4]),
        }
        radiance_cfg = {
            "use_siren": args.model.radiance.get(
                "use_siren", args.model.get("use_siren", False)
            ),
            "embed_multires": args.model.radiance.get("embed_multires", -1),
            "embed_multires_view": args.model.radiance.get("embed_multires_view", -1),
            "use_view_dirs": args.model.radiance.get("use_view_dirs", True),
            "D": args.model.radiance.get("D", 4),
            "W": args.model.radiance.get("W", 256),
            "skips": args.model.radiance.get("skips", []),
        }

    model_config = {
        "use_nerfplusplus": args.model.get("outside_scene", "builtin") == "nerf++",
        "use_tinycuda": args.model.get("use_tinycuda", False),
        "sdf_rep": args.model.get("sdf_rep", "net"),
        "obj_bounding_radius": args.model.obj_bounding_radius,
        "W_geo_feat": args.model.get("W_geometry_feature", 256),
        "speed_factor": args.training.get("speed_factor", 1.0),
        "beta_init": args.training.get("beta_init", 0.1),
    }

    model_config["data_size"] = data_size
    model_config["surface_cfg"] = surface_cfg
    model_config["radiance_cfg"] = radiance_cfg
    model_config["oTh_cfg"] = args.oTh

    hA_cfg = {"key": "hA", "data_size": data_size, "mode": args.hA.mode}
    model_config["hA_cfg"] = hA_cfg

    model = VolSDFHoi(**model_config)

    ## render_kwargs
    max_radius = kwargs.get("cam_norm", args.data.scale_radius)
    render_kwargs_train = {
        "near": max_radius - args.model.obj_bounding_radius,  # args.data.near,
        "far": max_radius + args.model.obj_bounding_radius,  # args.data.far,
        "batched": True,
        "perturb": args.model.get(
            "perturb", True
        ),  # config whether do stratified sampling
        "white_bkgd": args.model.get("white_bkgd", False),
        "max_upsample_steps": args.model.get("max_upsample_iter", 5),
        "use_nerfplusplus": args.model.get("outside_scene", "builtin") == "nerf++",
        "obj_bounding_radius": args.model.obj_bounding_radius,
        "N_samples": args.model.get("N_samples", 128),
        "calc_normal": True,
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test["rayschunk"] = args.data.val_rayschunk
    render_kwargs_test["perturb"] = False
    render_kwargs_test["calc_normal"] = True

    devices = kwargs.get("device", args.device_ids)
    trainer = Trainer(model, devices, batched=render_kwargs_train["batched"], 
                      args=args, test_mode=kwargs.get("test_mode", False))

    return model, trainer, render_kwargs_train, render_kwargs_test
