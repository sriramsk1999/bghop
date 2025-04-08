# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import mesh_utils, hand_utils
from einops import rearrange


class BaseHandField(nn.Module):
    def __init__(
        self,
        cfg,
        side,
    ) -> None:
        super().__init__()
        self.hand_wrapper = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean, side=side)
        field = cfg.get("field", "coord")
        field2ndim = {
            "coord": 45,
            "distance": 20,
            "none": 0,
        }
        self.ndim = field2ndim[field]
        self.cfg = cfg

    def forward_distance(self, hA, H, nXyz, rtn_wrist=True):
        N = len(hA)
        # nXyz = mesh_utils.create_sdf_grid(N, H, lim, 'zyx', device=device) # (B, H, H, H, 3) in range of (-1, 1)
        nXyz = rearrange(nXyz, "n d h w c -> n c d h w")  # (B, 3, H, H, H)
        nXyz = F.interpolate(nXyz, size=(H, H, H), mode="trilinear")
        nXyz = rearrange(nXyz, "n c d h w -> n 1 (d h w) c")  # (B,1, DHW, 3)

        _, hJoints = self.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=hA)
        nJoints = mesh_utils.apply_transform(hJoints, nTh)  # (N, J, 3)
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

    @torch.enable_grad()
    def grid2pose_sgd(self, jsPoints_gt, opt_mode="lbfgs", field="coord"):
        print("gradient descent to extract hand pose")
        N = len(jsPoints_gt)
        rtn = defaultdict(list)
        hA = nn.Parameter(self.hand_wrapper.hand_mean.repeat(N, 1))
        if opt_mode == "lbfgs":
            opt = torch.optim.LBFGS([hA], lr=0.1)
            T = 20
        elif opt_mode == "adam":
            opt = torch.optim.Adam([hA], lr=1e-2)
            T = 1000
        H = jsPoints_gt.shape[-1]

        hA_list = [hA.cpu().detach().clone()]
        # mini SGD to visualize SGD
        for i in range(T):

            def closure():
                opt.zero_grad()
                jsPoints = self.pose2grid(hA, H, field=field, tsdf=self.cfg.tsdf_hand)
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
        return hA.detach(), hA_list, rtn

    def pose2grid(self, hA, H, nXyz=None, field="coord", tsdf=None):
        N = len(hA)
        if nXyz is None:
            lim = self.cfg.side_lim
            nXyz = mesh_utils.create_sdf_grid(N, H, lim, device=hA.device)
        jsPoints = self(hA, H, nXyz, field=field, rtn_wrist=False)
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
        return self.forward_distance(hA, H, nXyz, rtn_wrist=rtn_wrist)


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
