# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import mesh_utils, hand_utils, geom_utils
from einops import rearrange


class BaseHandField(nn.Module):
    def __init__(
        self,
        cfg,
        side,
    ) -> None:
        super().__init__()
        cfg.flat_hand_mean = cfg.get("flat_hand_mean", True)
        self.hand_wrapper = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean, side=side)
        field = cfg.get("field", "coord")
        field2ndim = {
            "coord": 45,
            "distance": 20,
            "none": 0,
        }
        self.ndim = field2ndim[field]
        self.cfg = cfg

    def forward_distance(self, hA, H, nXyz, rtn_wrist=True, **kwargs):
        N = len(hA)
        # nXyz = mesh_utils.create_sdf_grid(N, H, lim, 'zyx', device=device) # (B, H, H, H, 3) in range of (-1, 1)
        nXyz = rearrange(nXyz, "n d h w c -> n c d h w")  # (B, 3, H, H, H)
        nXyz = F.interpolate(nXyz, size=(H, H, H), mode="trilinear")
        nXyz = rearrange(nXyz, "n c d h w -> n 1 (d h w) c")  # (B,1, DHW, 3)

        _, hJoints = self.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
        if kwargs["nTh_left"] is None:
            nJoints = mesh_utils.apply_transform(hJoints, nTh)  # (N, J, 3)
        else: # transform left hand distance field to normalized right hand coordinate frame
            nJoints = mesh_utils.apply_transform(hJoints, kwargs["nTh_left"][:, 0])  # (N, J, 3)
        nJoints = nJoints.unsqueeze(2)  # (N, J, 1, 3)

        nDist_square = ((nXyz - nJoints) ** 2).sum(-1)  # (N, J, DHW)
        nDist_square = rearrange(
            nDist_square, "n j (d h w) -> n j d h w", d=H, h=H, w=H
        )  # (N, J, D, H, W)

        # zero out jsPoints when all hA is zero
        zero_mask = (hA != 0).sum(
            -1,
        )  # (N, 1)  # all 0
        zero_mask = (zero_mask == 0).float()
        nDist_square = nDist_square * (1 - zero_mask.reshape(N, 1, 1, 1, 1))
        if not rtn_wrist:
            nDist_square = nDist_square[:, 1:]  # largest: 9.38
        return nDist_square

    def forward_coord_dist(self, hA, H, nXyz, rtn_wrist=True):
        # deprecated
        jsPoints = self.forward_coord(
            hA, H, nXyz, rtn_wrist=rtn_wrist
        )  # (N, J*3, D, H, W)
        jsPoints = rearrange(
            jsPoints, "n (j c) d h w -> n j c d h w", c=3
        )  # (N, J, DHW, 3)
        nDist = (jsPoints**2).sum(2)
        return nDist

    def forward_coord(self, hA, H, nXyz, rtn_wrist=True):
        nXyz = rearrange(nXyz, "n d h w c -> n c d h w")  # (B, 3, H, H, H)
        nXyz = F.interpolate(nXyz, size=(H, H, H), mode="trilinear")
        nXyz = rearrange(nXyz, "n c d h w -> n (d h w) c")  # (B, DHW, 3)
        jsPoints = hand_utils.transform_nPoints_to_js(
            self.hand_wrapper, hA, nXyz, iso=True
        )
        jsPoints = rearrange(jsPoints, "n (d h w) j c -> n (j c) d h w", d=H, h=H, w=H)

        jsPoints = self.mask_hA(jsPoints, hA)
        if not rtn_wrist:
            jsPoints = jsPoints[:, 3:]

        return jsPoints

    def mask_hA(self, jsPoints, hA):
        N = len(hA)
        legacy = self.cfg.get("legacy_cfg_hand", False)
        if not legacy:
            # zero out jsPoints when all hA is zero
            zero_mask = (hA != 0).sum(
                -1,
            )  # (N, 1)  # all 0
            zero_mask = (zero_mask == 0).float()
            jsPoints = jsPoints * (1 - zero_mask.reshape(N, 1, 1, 1, 1))
        return jsPoints

    def forward(
        self,
    ):
        return

    @torch.no_grad()
    def find_nTh_left(self, left_in_right_distance_field, hA_left, field):
        """
        The model is trained to predict the distance fields of both hands
        in the right hand's coordinate frame. Thus, while generating,
        we need to find the transform that best aligns the left hand
        to the right hand's coordinate frame.
        """
        def unravel_index(index, shape):
            idx = index
            coords = []
            for dim_size in reversed(shape):
                coords.append(idx % dim_size)
                idx = idx // dim_size
            return tuple(reversed(coords))  # Reverse to match original dimension order

        def find_rigid_transform(src, dst):
            assert src.shape == dst.shape and src.shape[1] == 3, "Inputs must be (N, 3)"

            # Compute centroids
            centroid_src = src.mean(dim=0)  # Shape: (3,)
            centroid_dst = dst.mean(dim=0)  # Shape: (3,)

            # Center the points
            src_centered = src - centroid_src  # Shape: (20, 3)
            dst_centered = dst - centroid_dst  # Shape: (20, 3)

            # Compute covariance matrix
            H = src_centered.T @ dst_centered  # Shape: (3, 3)

            # SVD for rotation
            U, S, Vt = torch.linalg.svd(H)  # U, Vt: (3, 3)
            R = Vt.T @ U.T  # Initial rotation

            # Fix reflection (ensure proper rotation)
            if torch.det(R) < 0:
                Vt[:, 2] *= -1  # Flip last column
                R = Vt.T @ U.T

            # Compute translation
            t = centroid_dst - R @ centroid_src  # Shape: (3,)

            return R, t  # R: (3, 3), t: (3,)

        H = left_in_right_distance_field.shape[-1]
        N = len(hA_left)
        nJoints = left_in_right_distance_field.shape[1]
        left_in_left_distance_field = self.pose2grid(hA_left, H, field=field, tsdf=self.cfg.tsdf_hand)
        lim = self.cfg.side_lim
        nXyz = mesh_utils.create_sdf_grid(N, H, lim, device=hA_left.device)
        n_leftTh_left = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA_left)

        # Get the coords of voxels with minimum distances to joints
        # lIl - left in left coord frame
        # lIr - left in right coord frame
        min_lIr_coords, min_lIl_coords = [], []
        for j in range(N):
            for i in range(nJoints):
                flat_idx = torch.argmin(left_in_right_distance_field[j][i])
                min_lIr_coords.append(unravel_index(flat_idx.item(), left_in_right_distance_field[j][i].shape))

                flat_idx = torch.argmin(left_in_left_distance_field[j][i])
                min_lIl_coords.append(unravel_index(flat_idx.item(), left_in_left_distance_field[j][i].shape))

        min_lIr_coords = torch.tensor(min_lIr_coords).reshape(N, nJoints, 3)
        min_lIl_coords = torch.tensor(min_lIl_coords).reshape(N, nJoints, 3)
        batch_idx = torch.arange(N)[:, None].expand(N, nJoints)

        left_in_right_pts = nXyz[batch_idx, min_lIr_coords[:, :, 0], min_lIr_coords[:, :, 1], min_lIr_coords[:, :, 2]]
        left_in_left_pts = nXyz[batch_idx, min_lIl_coords[:, :, 0], min_lIl_coords[:, :, 1], min_lIl_coords[:, :, 2]]

        nTh_left = []
        for i in range(N):
            R, t = find_rigid_transform(left_in_left_pts[i], left_in_right_pts[i])
            nTn_left_ = geom_utils.rt_to_homo(R, t)
            nTh_left_ = nTn_left_ @ n_leftTh_left[i]
            nTh_left.append(nTh_left_)

        return torch.stack(nTh_left)[:, None]


    @torch.enable_grad()
    def grid2pose_sgd(self, jsPoints_gt, opt_mode="lbfgs", field="coord", is_left=False):
        print("gradient descent to extract hand pose")
        N = len(jsPoints_gt)
        rtn = defaultdict(list)
        hA = nn.Parameter(self.hand_wrapper.hand_mean.repeat(N, 1))
        params = [hA]
        if is_left:
            nTh_left = self.find_nTh_left(jsPoints_gt, hA, field)
        else:
            nTh_left = None

        if opt_mode == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=0.1)
            T = 20
        elif opt_mode == "adam":
            opt = torch.optim.Adam(params, lr=1e-2)
            T = 1000
        H = jsPoints_gt.shape[-1]

        hA_list = [hA.cpu().detach().clone()]
        # mini SGD to visualize SGD
        for i in range(T):

            def closure():
                opt.zero_grad()
                jsPoints = self.pose2grid(hA, H, field=field, tsdf=self.cfg.tsdf_hand, nTh_left=nTh_left)
                grid_loss = 1e4 * F.mse_loss(jsPoints, jsPoints_gt)
                reg_loss = 0.1 * F.mse_loss(
                    hA, self.hand_wrapper.hand_mean.repeat(N, 1)
                )
                loss = grid_loss + reg_loss
                loss.backward()
                grad = hA.grad
                rtn["grad"].append(grad.cpu().detach().clone().abs().mean(0))
                rtn["field"] = jsPoints
                return loss

            opt.step(closure)
            if i % T // 10 == 0 or i == T - 1:
                hA_list.append(hA.cpu().detach().clone())

        return hA.detach(), hA_list, rtn, nTh_left

    def pose2grid(self, hA, H, nXyz=None, field="coord", tsdf=None, nTh_left=None):
        N = len(hA)
        if nXyz is None:
            lim = self.cfg.side_lim
            nXyz = mesh_utils.create_sdf_grid(N, H, lim, device=hA.device)
        jsPoints = self(hA, H, nXyz, field=field, rtn_wrist=False, nTh_left=nTh_left)
        if tsdf is not None:
            jsPoints = jsPoints.clamp(-tsdf, tsdf)
        return jsPoints


class CoordField(BaseHandField):
    def __init__(self, cfg, side, **kwargs) -> None:
        super().__init__(cfg, side)
        self.ndim = 45

    def forward(self, hA, H, nXyz, rtn_wrist=True, **kwargs):
        return self.forward_coord(hA, H, nXyz, rtn_wrist=rtn_wrist)


class DistanceField(BaseHandField):
    def __init__(self, cfg, side) -> None:
        super().__init__(cfg, side)
        self.ndim = 20

    def forward(self, hA, H, nXyz, rtn_wrist=True, **kwargs):
        return self.forward_distance(hA, H, nXyz, rtn_wrist=rtn_wrist, **kwargs)


class Identity(BaseHandField):
    def __init__(self, cfg, side) -> None:
        super().__init__(cfg, side)
        self.ndim = 0

    def forward(self, hA, H, *args, **kwargs):
        """
        :param hA: pose in shape of (N, 45)
        :param H
        :return: (N, C=48, D, H, W)
        """
        C = 0
        N = len(hA)
        device = hA.device
        return torch.zeros([N, C, H, H, H], device=device)


def build_hand_field(field, cfg, side="right") -> BaseHandField:
    str2field = {
        "distance": DistanceField,
        "coord": CoordField,
        "none": Identity,
        "base": BaseHandField,
    }
    return str2field[field](cfg, side=side)
