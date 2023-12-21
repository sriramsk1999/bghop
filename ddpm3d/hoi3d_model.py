# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
# 3D equivalent of ddpm2d/models/glide.py
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.distributed import rank_zero_only

from .base import BaseModule
from .models.text2sdf import Text2GFUNet
from .models.autoencoder import VQModelWrapper, VQModel
from .utils.ddpm_util import create_gaussian_diffusion
from utils.viz_utils import Visualizer
from utils.hand_utils import build_hand_field
from jutils import model_utils, mesh_utils


class LatentObjIdtyHand(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        # initialize ae
        self.ae = model_utils.instantiate_from_config(cfg.first_stage, cfg=cfg)
        self.hand_cond = build_hand_field(cfg.field, cfg)

        if isinstance(self.ae, VQModel):
            self.ae = VQModelWrapper(self.ae)
        model_utils.freeze(self.ae)
        side_x = self.cfg.side_x // self.ae.downsample
        self.latent_dim = self.ae.embed_dim
        ndim = self.latent_dim + self.hand_cond.ndim
        self.template_size = [ndim, side_x, side_x, side_x]

        # initialize diffusion
        options = self.cfg.model.diffuse
        self.diffusion = create_gaussian_diffusion(
            steps=options.diffusion_steps,
            noise_schedule=options.noise_schedule,
            timestep_respacing=options.timestep_respacing,
        )
        self.glide_options = options

        self.cfg.model.unet.params["in_channels"] = ndim
        self.cfg.model.unet.params["out_channels"] = ndim
        glide_model = Text2GFUNet(self.cfg)
        glide_model.tokenizer = glide_model.text_cond_model.tknz_fn
        self.glide_model = glide_model

        self.viz = Visualizer(cfg, self.log_dir)

        self.mean = 0
        std = [
            self.cfg.std_latent,
        ] * self.latent_dim + [
            self.cfg.std_hand,
        ] * self.hand_cond.ndim
        self.register_buffer("std", torch.FloatTensor(std).reshape(1, -1, 1, 1, 1))

    def set_inputs(self, batch, batch_idx=0, **kwargs):
        return self.ae.set_inputs(batch, **kwargs)

    def norm_latents(self, x):
        if self.cfg.get("norm_latent", False):
            x = (x - self.mean) / self.std
        return x

    def unnorm_latents(self, x):
        if self.cfg.get("norm_latent", False):
            x = x * self.std + self.mean
        return x

    def encode(self, x, batch):
        image = self.ae.encode(x)
        hand = self.hand_cond(
            batch["hA"], image.shape[-1], batch["nXyz"], rtn_wrist=False
        )
        if self.cfg.tsdf_hand is not None:
            hand = hand.clamp(min=-self.cfg.tsdf_hand, max=self.cfg.tsdf_hand)
        image = torch.cat([image, hand], dim=1)
        image = self.norm_latents(image)
        batch["hand"] = image[:, self.latent_dim :]
        return image

    @torch.no_grad()
    def decode(self, z):
        z = self.unnorm_latents(z)

        obj = z[:, : self.latent_dim]
        hand = z[:, self.latent_dim :]

        x = self.ae.decode(obj)  
        rtn = {
            "image": x,
            "hand": hand,  
        }
        return rtn

    def step(self, batch, batch_idx):
        """change SDF scale to -1, 1"""
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion
        text, _ = batch["text"], batch["image"]

        inputs = self.encode(batch["image"], batch)

        batch_size = len(inputs)
        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (batch_size,), device=device
        )
        noise = torch.randn([batch_size] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(
            inputs,
            timesteps,
            noise=noise,
        ).to(device)
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            text=text,
        )[:, 0 : self.template_size[0]]
        epsilon = model_output
        loss = F.mse_loss(epsilon, noise.to(device).detach())
        return loss, {"loss": loss}

    @torch.no_grad()
    def decode_samples(self, sample, offset):
        """Extract Object surface~~
        :param tensor: (N, 1, D, H, W)
        :return: Meshes
        """
        x = sample["image"]
        sdf = self.ae.unnorm(x)

        N = len(sdf)
        jObj = mesh_utils.batch_grid_to_meshes(
            sdf, N, half_size=self.cfg.side_lim, offset=offset
        )
        jObj.textures = mesh_utils.pad_texture(jObj)

        return {
            "jObj": jObj,
            "raw_sdf": sdf,
        }

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):
        dec = self.decode_samples(samples, batch["offset"])
        # offset = batch['offset']
        jObj = dec["jObj"]
        self.vis_meshes(
            jObj,
            f"{pref}_sample_jObj",
            log,
            step,
            text=[" ".join(e.split(" ")[6:]) for e in batch["text"]],
        )

        nTw = mesh_utils.get_nTw(jObj)

        if "hand" in samples:
            print("vis samples")
            hA, hA_list, rtn = self.hand_cond.grid2pose_sgd(
                samples["hand"], field=self.cfg.field
            )
            self.viz.render_hA_traj(hA_list, f"{pref}_hA_traj", log, step, nTw.device)
            self.viz.render_hoi(jObj, hA, f"{pref}_jHoi", log, step)
            self.viz.render_hand(hA, f"{pref}_hHand", log, step)

    def get_model_kwargs(
        self, device, batch_size, val_batch, text="", uncond_image=False
    ):
        glide_model = self.glide_model
        # import pdb; pdb.set_trace()
        uncond = glide_model.get_text_cond([""] * batch_size)
        cond = glide_model.get_text_cond(text)
        if uncond is None:
            text = cond
        else:
            text = torch.cat([cond, uncond], dim=0).to(device)

        if "nXyz" not in val_batch:
            # N = len(val_batch['hA'])
            N = batch_size
            reso = self.cfg.side_x
            lim = self.cfg.side_lim
            nXyz = mesh_utils.create_sdf_grid(N * 2, reso, lim, device=device)
        else:
            nXyz = torch.cat([val_batch["nXyz"]] * 2, dim=0)

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            text=text,
            nXyz=nXyz,
        )
        return model_kwargs
