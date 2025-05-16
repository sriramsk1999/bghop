from hydra import main

import math
import numpy as np

# import tinycudann as tcnn
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torch.nn.functional as F
from jutils import mesh_utils

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class TinyImplicitSurface(nn.Module):
    def __init__(self, 
                 W_geo_feat=256,
                 W=256,
                 D=8,
                 skips=[4],
                 input_ch=3,
                 radius_init=1.0,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 use_grid=True,
                 encoding_config={}, 
                 ) -> None:
        super().__init__()
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.skips = skips
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        self.use_grid = use_grid
        if use_grid:
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=dict(encoding_config),
            )
            self.grid_dim = self.encoding.n_output_dims
        else:
            self.grid_dim = 10
        self.embed_fn, input_ch = get_embedder(embed_multires)
        input_ch += self.grid_dim

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D+1):
            # decide out_dim
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = W
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first = (l==0))
                else:
                    # NOTE: beta=100 is important! Otherwise, the initial output would all be > 10, and there is not initial sphere.
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init and not use_siren:
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor, return_h = False):
        x_dim = x.shape[:-1]
        _3 = x.shape[-1]
        if self.use_grid:
            feature = self.encoding(x.reshape(-1, _3)).reshape(*x_dim, -1).to(x.dtype)
        else:
            feature = torch.zeros_like(x[..., :1].expand(*x_dim, self.grid_dim))
        pe = self.embed_fn(x)
        x = h =  torch.cat((pe, feature), dim=-1)

        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)
            nabla = autograd.grad(
                implicit_surface_val,
                x,
                torch.ones_like(implicit_surface_val, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h
        


class ExplicitGrid(nn.Module):
    def __init__(self,
                 dim=1,
                 half_size=1,
                 reso=64,
                 border='extra',
                 grids = None,
                 ):    
        super().__init__()
        self.half_size = half_size #  / 2
        self.border = border
        self.dim = dim
        if grids is None:
            xyz = mesh_utils.create_sdf_grid(1, reso, half_size)  # (B, H, H, H, 3)
            if dim == 1:
                dist = (torch.norm(xyz, dim=-1).unsqueeze(1) - half_size / 4)
                print(dist.shape)
            else:
                dist = xyz.permute(0, 4, 1, 2, 3)  # (B, 3, H, H, H)
            self.grids = nn.Parameter(dist.reshape(1, dim, reso, reso, reso))

        else:
            self.grids = nn.Parameter(grids.reshape(1, dim, reso, reso, reso).transpose(-1, -3))

    def forward(self, x: torch.Tensor, *args, return_h = False, order='zyx', **kwargs,):
        """

        Args:
            x (torch.Tensor): (N?, P, 3)
            return_h (bool, optional): _description_. Defaults to False.
        """
        # print('xminmax', x.min(), x.max())
        # TODO: 
        # print('double check scale of your grids, where to put onTn?')
        x_dim = x.shape[:-1]
        _3 = x.shape[-1]
        xyz = x.reshape(1, -1, 1, 1, _3)

        if order == 'zyx':
            grids = self.grids.transpose(-1, -3)
            xyz = xyz.flip(-1)
        else:
            grids = self.grids
        
        xyz = xyz / self.half_size # -1, 1
        value = F.grid_sample(grids, xyz, align_corners=True, mode='bilinear', padding_mode='border')
        # value = grid_sample_3d(grids, xyz)

        if self.border == 'extra':
            with torch.no_grad():
                extra_dist_in_a = (torch.abs(xyz) - 1).clamp_(min=0)
                extra_dist_in_a = (extra_dist_in_a**2).sum(dim=-1).sqrt()  # (N, D, H, W)
                extra_dist = extra_dist_in_a * self.half_size
            value = value + extra_dist

        if order == 'zyx':
            pass
        
        value = value.reshape(*x_dim, self.dim)
        if self.dim == 1:
            value = value.squeeze(-1)
        if return_h:
            return value, None
        return value

    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)

            detached_x = x.detach().requires_grad_(True)
            implicit_surface_val2, _ = self.forward(detached_x, return_h=True)
            nabla = torch.zeros_like(x)
            # nabla = autograd.grad(
            #     implicit_surface_val2,
            #     detached_x,
            #     torch.ones_like(implicit_surface_val2, device=x.device),
            #     create_graph=has_grad,
            #     retain_graph=has_grad,
            #     only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
        return implicit_surface_val, nabla, h


class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256,
                 D=8,
                 skips=[4],
                 W_geo_feat=256,
                 input_ch=3,
                 radius_init=1.0,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 ):
        """
        W_geo_feat: to set whether to use nerf-like geometry feature or IDR-like geometry feature.
            set to -1: nerf-like, the output feature is the second to last level's feature of the geometry network.
            set to >0: IDR-like ,the output feature is the last part of the geometry network's output.
        """
        super().__init__()
        # occ_net_list = [
        #     nn.Sequential(
        #         nn.Linear(input_ch, W),
        #         nn.Softplus(),
        #     )
        # ] + [
        #     nn.Sequential(
        #         nn.Linear(W, W),
        #         nn.Softplus()
        #     ) for _ in range(D-2)
        # ] + [
        #     nn.Linear(W, 1)
        # ]
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D+1):
            # decide out_dim
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = W
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first = (l==0))
                else:
                    # NOTE: beta=100 is important! Otherwise, the initial output would all be > 10, and there is not initial sphere.
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init and not use_siren:
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def pretrain_hook(self, configs={}):
        configs['target_radius'] = self.radius_init
        # TODO: more flexible, bbox-like scene bound.
        configs['obj_bounding_size'] = self.obj_bounding_size.item()
        if self.geometric_init and self.use_siren and not self.is_pretrained:
            pretrain_siren_sdf(self, **configs)
            self.is_pretrained = ~self.is_pretrained
            return True
        return False

    def forward(self, x: torch.Tensor, return_h = False):
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)

            detached_x = x.detach().requires_grad_(True)
            implicit_surface_val2, _ = self.forward(detached_x, return_h=True)
            nabla = autograd.grad(
                implicit_surface_val2,
                detached_x,
                torch.ones_like(implicit_surface_val2, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h


def pretrain_siren_sdf(
    implicit_surface: ImplicitSurface,
    num_iters=5000, lr=1.0e-4, batch_points=5000, 
    target_radius=0.5, obj_bounding_size=3.0,
    logger=None):
    #--------------
    # pretrain SIREN-sdf to be a sphere, as in SIREN and Neural Lumigraph Rendering
    #--------------
    from tqdm import tqdm
    from torch import optim
    device = next(implicit_surface.parameters()).device
    optimizer = optim.Adam(implicit_surface.parameters(), lr=lr)
    
    with torch.enable_grad():
        for it in tqdm(range(num_iters), desc="=> pretraining SIREN..."):
            pts = torch.empty([batch_points, 3]).uniform_(-obj_bounding_size, obj_bounding_size).float().to(device)
            sdf_gt = pts.norm(dim=-1) - target_radius
            sdf_pred = implicit_surface.forward(pts)
            
            loss = F.l1_loss(sdf_pred, sdf_gt, reduction='mean')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if logger is not None:
                logger.add('pretrain_siren', 'loss_l1', loss.item(), it)


class TinyRadianceNet(nn.Module):
    def __init__(self, 
        W_geo_feat=256,
        D=4,
        W=256,
        skips=[],
        embed_multires=6,
        embed_multires_view=4,
        use_view_dirs=True,
        weight_norm=True,
        use_siren=False,
        encoding_config={},
    ) -> None:
        super().__init__()

        input_ch_pts = 3
        input_ch_views = 3
        self.geo_feat_dim = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs

        self.embed_fn_view = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=dict(encoding_config),
            # {
            #     "otype": "SphericalHarmonics",
            #     "degree": 4,
            # },
        )
        if use_view_dirs:
            in_dim = self.embed_fn_view.n_output_dims \
                + 3 + self.geo_feat_dim
        else:
            in_dim = self.geo_feat_dim

        in_dim_0 = in_dim
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
            
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor):
        """
        Args:
            x: position in shape of (..., 3)
            view_dirs: in shape of (..., 3)
        """
        x_dim = view_dirs.shape[:-1]
        _3 = view_dirs.shape[-1]
        D = geometry_feature.shape[-1]

        # calculate radiance field
        # x = self.embed_fn(x)
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs.reshape(-1, _3)).to(x.dtype)
            radiance_input = torch.cat([
                view_dirs, 
                normals.reshape(-1, _3), 
                geometry_feature.reshape(-1, D)
            ], dim=-1)
        else:
            radiance_input = geometry_feature.reshape(-1, _3)
        
        h = radiance_input        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)

        h = h.reshape(*x_dim, -1)
        return h        


class RadianceNet(nn.Module):
    def __init__(self,
        D=4,
        W=256,
        skips=[],
        W_geo_feat=256,
        embed_multires=6,
        embed_multires_view=4,
        use_view_dirs=True,
        weight_norm=True,
        use_siren=False,):
        super().__init__()
        
        input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        if use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_pts + input_ch_views + 3 + W_geo_feat
        else:
            in_dim_0 = input_ch_pts + W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate radiance field
        x = self.embed_fn(x)
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs)
            radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        else:
            radiance_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


# modified from https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=3, multires=-1, multires_view=-1, output_ch=4, skips=[4], use_view_dirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_view_dirs = use_view_dirs

        self.embed_fn, input_ch = get_embedder(multires, input_dim=input_ch)
        self.embed_fn_view, input_ch_view = get_embedder(multires_view, input_dim=input_ch_view)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_view_dirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        input_pts = self.embed_fn(input_pts)
        input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_view_dirs:
            sigma = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = F.relu_(h)

            rgb = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]
        
        rgb = torch.sigmoid(rgb)
        return sigma.squeeze(-1), rgb


class ScalarField(nn.Module):
    # TODO: should re-use some feature/parameters from implicit-surface net.
    def __init__(self, input_ch=3, W=128, D=4, skips=[], init_val=-2.0):
        super().__init__()
        self.skips = skips
        
        pts_linears = [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in skips 
             else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        for linear in pts_linears:
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)


        self.pts_linears = nn.ModuleList(pts_linears)
        self.output_linear = nn.Linear(W, 1)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, init_val)

    def forward(self, x: torch.Tensor):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        out = self.output_linear(h).squeeze(-1)
        return out


def get_optimizer(args, model, posenet, focalnet):    
    # if isinstance(args.training.lr, numbers.Number):
    #     optimizer = optim.AdamW(model.parameters(), lr=args.training.lr)
    # elif isinstance(args.training.lr, dict):

    #     lr_dict = args.training.lr
    #     default_lr = lr_dict.pop('default')
        
    #     param_groups = []
    #     select_params_names = []
    #     for name, lr in lr_dict.items():
    #         if name in model._parameters.keys():
    #             select_params_names.append(name)
    #             param_groups.append({
    #                 'params': getattr(model, name),
    #                 'lr': lr
    #             })
    #         elif name in model._modules.keys():
    #             select_params_names.extend(["{}.{}".format(name, param_name) for param_name, _ in getattr(model, name).named_parameters()])
    #             param_groups.append({
    #                 'params': getattr(model, name).parameters(),
    #                 'lr': lr
    #             })
    #         else:
    #             raise RuntimeError('wrong lr key:', name)

    #     # NOTE: parameters() is just calling named_parameters without returning name.
    #     other_params = [param for name, param in model.named_parameters() if name not in select_params_names]
    #     param_groups.insert(0, {
    #         'params': other_params,
    #         'lr': default_lr
    #     })
    # else:
        # raise NotImplementedError
    lr_dict = args.training.lr
    model_but_oTh = []
    texture = []
    oTh = []

    for name, param in model.named_parameters():
        if 'oTh' in name or 'h_leftTh' in name:
            print('another param', name)
            oTh.append(param)
        elif 'uv_text' in name:
            print('another param', name)
            texture.append(param)
        else:
            print(name)
            model_but_oTh.append(param)
    param_groups = [
        {'params': posenet.parameters(), 'lr': lr_dict['pose']},
        {'params': model_but_oTh, 'lr': lr_dict['model']},
        {'params': oTh, 'lr': lr_dict['oTh']},
        {'params': texture, 'lr': lr_dict['text']},
        {'params': focalnet.parameters(), 'lr': lr_dict['focal'], 
        }            
    ]
    
    optimizer = optim.AdamW(params=param_groups)
    return optimizer


def CosineAnnealWarmUpSchedulerLambda(total_steps, warmup_steps, min_factor=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        """
        modified from https://github.com/Totoro97/NeuS/blob/main/exp_runner.py
        """
        if epoch < warmup_steps:
            learning_factor = epoch / warmup_steps
        else:
            learning_factor = (np.cos(np.pi * ((epoch - warmup_steps) / (total_steps - warmup_steps))) + 1.0) * 0.5 * (1-min_factor) + min_factor
        return learning_factor
    return lambda_fn


def ExponentialSchedulerLambda(total_steps, min_factor=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        t = np.clip(epoch / total_steps, 0, 1)
        learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor
    return lambda_fn


def get_scheduler(args, optimizer, last_epoch=-1):
    stype = args.training.scheduler.type
    if stype == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                args.training.scheduler.milestones, 
                gamma=args.training.scheduler.gamma, 
                last_epoch=last_epoch)
    elif stype == 'warmupcosine':
        # NOTE: this do not support per-parameter lr
        # from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        # scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer, 
        #     args.training.num_iters, 
        #     max_lr=args.training.lr, 
        #     min_lr=args.training.scheduler.setdefault('min_lr', 0.1*args.training.lr), 
        #     warmup_steps=args.training.scheduler.warmup_steps, 
        #     last_epoch=last_epoch)
        # NOTE: support per-parameter lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            CosineAnnealWarmUpSchedulerLambda(
                total_steps=args.training.num_iters, 
                warmup_steps=args.training.scheduler.warmup_steps, 
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            ),
            last_epoch=last_epoch)
    elif stype == 'exponential_step':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            ExponentialSchedulerLambda(
                total_steps=args.training.num_iters,
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            )
        )
    else:
        raise NotImplementedError
    return scheduler


def test():
    """
    test siren-sdf pretrain
    """
    def eikonal_loss(nabla):
        nablas_norm = torch.norm(nabla, dim=-1,)
        loss = F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')        
        return loss
        
    import time
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/model/tiny.yaml')

    device = 'cuda:0'
    surface = TinyImplicitSurface(cfg.W_geometry_feature, **cfg.surface).to(device)
    radiance = TinyRadianceNet(cfg.W_geometry_feature, **cfg.radiance).to(device)

    opt = optim.AdamW(surface.parameters(), lr=1e-4)
    R = 64
    B = 1
    inp = torch.randn([B, R, 1, 3], device=device).clamp(-0.1, 0.1)
    inp2 = torch.randn([B, R, 1, 3], device=device)
    view = torch.randn([B, R,  1, 3], device=device)
    normals = torch.randn([B, R, 1, 3], device=device)

    t = time.time()
    for _ in range(1000):
        opt.zero_grad()
        sdf, nabla, out = surface.forward_with_nablas(inp, True)
        # out = radiance(inp, view, normals, out)

        loss = eikonal_loss(nabla)
        print(loss)
        loss.backward()
        opt.step()

    # clear cuda
    torch.cuda.empty_cache()
    # print('cuda memory:', torch.cuda.memory_allocated(device=device) / 1024**2, 'MB')
    cfg = OmegaConf.load('configs/model/small.yaml')
    W_geo_feat = cfg.W_geometry_feature
    input_ch = 3
    obj_bounding_radius = cfg.obj_bounding_radius
    surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **cfg.surface).to(device)
    radiance = RadianceNet(
            W_geo_feat=W_geo_feat, **cfg.radiance).to(device)
    t = time.time()
    sdf, nabla, out = surface.forward_with_nablas(inp, True)
    out = radiance(inp, view, normals, out)
    _, nablas2, _ = surface.forward_with_nablas(inp2, True)
    
    nabla = torch.cat([nabla, nablas2], dim=-2)
    loss = eikonal_loss(nabla)
    loss = loss.backward()
    print('shape: ', sdf.shape, nabla.shape, out.shape)

    # print('forward time:', time.time() - t)
    # # count parameters
    # print('surface parameters:', sum(p.numel() for p in surface.parameters()))
    # print('radiance parameters:', sum(p.numel() for p in radiance.parameters()))
    # print('cuda memory:', torch.cuda.memory_allocated(device=device) / 1024**2, 'MB')
    return 


def vis_scheduler():
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    method_list = ['exponential_step', 'warmupcosine']
    opt = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=1)
    for method in method_list:
        cfg = OmegaConf.create(
            {'training': {'scheduler': 
                          {'type': method, 
                           'warmup_steps': 2500,  # 5% of total steps
                           'min_factor': 0.1},
                           'num_iters': 50000,}})
        scheduler = get_scheduler(cfg, opt)
        lr_list = []
        for i in range(cfg.training.num_iters):
            scheduler.step(i)
            lr_list.append(scheduler.get_lr()[0])
        plt.plot(lr_list, label=method)
    plt.legend(method_list)
    plt.savefig('/home/yufeiy2/scratch/result/vis/lr_scheduler.png')
    return 


@main('../configs/model', 'small')
def test_grid(cfg):
    save_dir = '/private/home/yufeiy2/scratch/result/vis/'
    reso = 64
    lim = side_lim = 1.5
    B = 1
    device = 'cuda:0'
    mesh_file = osp.join(save_dir, 'mesh.ply')
    
    surface_cfg = cfg.surface
    net = ImplicitSurface(**surface_cfg, ).to(device)

    mesh_util.extract_mesh(
        net, 
        N=reso,
        filepath=mesh_file,
        volume_size=side_lim*2,
    )
    jObj = mesh_utils.load_mesh(mesh_file).cuda()
    jObj.textures = mesh_utils.pad_texture(jObj, 'red')

    # queried output
    jXyz = mesh_utils.create_sdf_grid(B, reso, lim, 'zyx', device=device) # (B, H, H, H, 3) in range of (-1, 1)
    jXyz = jXyz.reshape(B, -1, 3)
    jSdf, nblas, h = net.forward_with_nablas(jXyz)  # (N, R, ?)
    jSdf = jSdf.reshape(B, 1, reso, reso, reso) # fuzzy!!!
    jSdf_mesh = mesh_utils.batch_grid_to_meshes(jSdf, 1, device=device, half_size=lim)
    jSdf_mesh.textures = mesh_utils.pad_texture(jSdf_mesh, 'blue')
    
    out = net(jXyz)
    print('jXyz', jXyz.shape, out.shape)
    out = net(jXyz.reshape(B, reso, reso, reso, 3))
    print('jXyz', jXyz.shape, out.shape)

    grid_net = ExplicitGrid(1, lim*2, reso, grids=jSdf).to(device)
    grids = grid_net(jXyz).reshape(B, 1, reso, reso, reso)

    grids_mesh = mesh_utils.batch_grid_to_meshes(grids, 1, device=device, half_size=lim)
    grids_mesh.textures = mesh_utils.pad_texture(grids_mesh)

    coord = plot_utils.create_coord(device, 1, )
    scene = mesh_utils.join_scene([jObj, jSdf_mesh, grids_mesh, coord])

    image_list = mesh_utils.render_geom_rot_v2(scene)
    image_utils.save_gif(image_list, osp.join(save_dir, 'grid'))

    print(grids - jSdf)
    return 

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.reshape(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

if __name__ == "__main__":
    from jutils import mesh_utils, image_utils, geom_utils, plot_utils
    from utils import io_util, mesh_util
    import os
    import os.path as osp
    # test()
    test_grid()

    # vis_scheduler()