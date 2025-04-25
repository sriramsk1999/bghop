# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

# modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
import numpy as np
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from ddpm3d.base import BaseModule
from jutils import model_utils
from glide_text2im.model_creation import create_gaussian_diffusion
from glide_text2im.gaussian_diffusion import _extract_into_tensor
from glide_text2im.respace import _WrappedModel
from utils.obj2text import Obj2Text
from utils.io_util import load_from_checkpoint


class SDLoss:
    def __init__(
        self,
        ckpt_path,
        cfg={},
        anneal_noise="constant",
        min_step=0.02,
        max_step=0.98,
        to_step=0.5,
        grad_clip_val=None,
        prediction_respacing=100,
        guidance_scale=4,
        grad_mode="noise",
        wgt_method="idty",
        prompt="a semantic segmentation of a hand grasping an object",
        **kwargs,
    ) -> None:
        super().__init__()
        self._warp = None
        self.min_ratio = min_step
        self.max_ratio = max_step
        self.to_ratio = min(to_step, max_step)
        self.min_step = 0
        self.max_step = 0
        self.grad_clip_val = grad_clip_val
        self.grad_mode = grad_mode
        self.anneal_noise = anneal_noise
        self.num_step = prediction_respacing
        self.guidance_scale = guidance_scale
        self.ckpt_path = ckpt_path
        self.model: BaseModule = None
        self.in_channles = 0
        self.alphas_bar = (
            None  #  self.scheduler.alphas_cumprod.to(self.device) # for convenience
        )
        self.cfg = cfg
        # TODO: change constant prompt
        self.wgt_method = wgt_method
        self.const_str = prompt
        self.reso = 64
        self.text_template = {}

    def to(self, device):
        self.model.to(device)

    def init_model(self, device="cuda"):
        print(self.ckpt_path)
        self.model = load_from_checkpoint(self.ckpt_path)
        self.model.eval()
        model_utils.freeze(self.model)
        self.unet = self.model.glide_model
        self.options = self.model.glide_options

        self.diffusion = create_gaussian_diffusion(
            steps=self.options["diffusion_steps"],
            noise_schedule=self.options["noise_schedule"],
            timestep_respacing=str(self.num_step),
        )
        self.min_step = int(self.min_ratio * self.num_step)
        self.max_step = int(self.max_ratio * self.num_step)
        self.alphas_bar = self.diffusion.alphas_cumprod
        self.in_channles = self.model.template_size[0]

        if self.grad_mode == "noise" and hasattr(self.model, "encode"):
            print("unfreeze")
            model_utils.unfreeze(self.model.ae)

        self.to(device)  # do this since loss is not a nn.Module?
        lib_name = self.model.cfg.get("lib", None)
        if lib_name is not None:
            lib_name = osp.join(
                self.model.cfg.environment.data_dir, f"lib/{lib_name}.json"
            )
        lib_name = lib_name if osp.exists(lib_name) else None
        self.text_template = Obj2Text(lib_name)

    def get_weight(self, t, shape, method: str):
        if method == "dream":
            w = 1 - _extract_into_tensor(self.alphas_bar, t, shape)
        elif method == "bell":
            w = (
                1 - _extract_into_tensor(self.alphas_bar, t, shape)
            ) ** 0.5 * _extract_into_tensor(np.sqrt(self.alphas_bar), t, shape)
        elif method.startswith("bell-p"):
            p = float(method[len("bell-p") :])
            w = (1 - _extract_into_tensor(self.alphas_bar, t, shape)) ** 0.5 * (
                _extract_into_tensor(np.sqrt(self.alphas_bar), t, shape) ** p
            )
        return w

    def schedule_max_step(self, it, min_value=None):
        method = self.anneal_noise
        min_ratio = self.to_ratio

        max_iter = self.num_step  # self.cfg.training.num_iters
        if method == "linear":
            # linearly anneal noise level from max_ratio to min_ratio
            max_ratio = self.max_ratio * (1 - it / max_iter) + min_ratio * (
                it / max_iter
            )
        elif method == "constant":
            max_ratio = self.max_ratio
        elif method == "adaptive":
            # linear interpolate: min: 0.25 if minvalue == -0.01, max: 0.75. if minvalue == -0.2
            a = -0.04 * 5
            b = -1
            alpha = (min_value - a) / (b - a)
            alpha = alpha.clamp(0, 1)
            max_ratio = self.min_ratio * (1 - alpha) + self.max_ratio * alpha
        else:
            raise NotImplementedError(f"Unknown anneal_noise method {method}")
        self.max_step = int(max_ratio * self.num_step)
        return self.max_step

    def _get_grad_in_noise_latents(
        self,
        batch,
        weight=1,
        t=None,
        noise=None,
        w_schdl="dream",
        debug=False,
        text=None,
        encode=True,
        **kwargs,
    ):
        device = batch["image"].device
        latents = batch["image"]
        guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
        if encode and hasattr(self.model, "encode"):
            latents = self.model.encode(latents, batch)
        if noise is None:
            noise = torch.randn_like(latents)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            latents_noisy = self.get_noisy_image(latents, t, noise)
            # pred noise
            model_fn = self.get_cfg_model_fn(self.unet, guidance_scale)
            noise_pred = self.get_pred_noise(model_fn, latents_noisy, t, batch, text[0])
            start_pred = self.diffusion.eps_to_pred_xstart(latents_noisy, noise_pred, t)
            if hasattr(self.model, "decode"):
                start_pred = self.model.decode(start_pred)["image"]

        # w(t), sigma_t^2
        w = self.get_weight(t, noise_pred.shape, w_schdl)
        w = w.to(device)

        grad = weight * w * (noise_pred - noise)

        targets = (latents - grad).detach()
        loss = (
            0.5
            * F.mse_loss(latents.float(), targets, reduction="sum")
            / latents.shape[0]
        )

        return loss, latents, {"start_pred": start_pred, "w": w}

    def _get_grad_in_xtart_onestep(
        self,
        batch,
        weight=1,
        t=None,
        noise=None,
        w_schdl="dream",
        debug=False,
        text=None,
        encode=True,
        **kwargs,
    ):
        latents = batch["image"]
        guidance_scale = self.guidance_scale
        if encode and hasattr(self.model, "encode"):
            latents = self.model.encode(latents, batch)
        if noise is None:
            noise = torch.randn_like(latents)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            latents_noisy = self.get_noisy_image(latents, t, noise)
            # pred noise
            model_fn = self.get_cfg_model_fn(self.unet, guidance_scale)
            noise_pred = self.get_pred_noise(model_fn, latents_noisy, t, batch, text[0])
            start_pred = self.diffusion.eps_to_pred_xstart(latents_noisy, noise_pred, t)
            start_pred_dict = {"image": start_pred}
            if hasattr(self.model, "decode"):
                start_pred_dict = self.model.decode(start_pred)
                start_pred = start_pred_dict["image"]

        w = self.get_weight(
            t,
            [
                1,
            ],
            w_schdl,
        ).item()

        loss = 0
        xa = batch["image"]
        xb = start_pred
        multi_scale = kwargs.get("multi_scale", 1)
        for i in range(multi_scale):
            loss = loss + weight * w * F.mse_loss(xa, xb)
            h = xa.shape[-1] // 2
            xa = F.adaptive_avg_pool3d(xa, h)
            xb = F.adaptive_avg_pool3d(xb, h)

        losses = {"diff_image": loss}
        grad = loss

        if "hand" in start_pred_dict:
            hand_pred = start_pred_dict["hand"]
            w_hand = kwargs.get("w_hand", 1)
            hand_loss = 0.5 * weight * w * w_hand * F.mse_loss(batch["hand"], hand_pred)
            grad += hand_loss
            losses["diff_hand"] = hand_loss

        if "hand_left" in start_pred_dict:
            hand_pred_left = start_pred_dict["hand_left"]
            w_hand = kwargs.get("w_hand", 1)
            hand_left_loss = 0.5 * weight * w * w_hand * F.mse_loss(batch["hand_left"], hand_pred_left)
            grad += hand_left_loss
            losses["diff_hand_left"] = hand_left_loss
        extras = {
            "start_pred": start_pred,
            "w": w,
            "noise_pred": noise_pred,
            "losses": losses,
        }
        if "hand" in start_pred_dict:
            extras["hand_pred"] = hand_pred
        if "hand_left" in start_pred_dict:
            extras["hand_pred"] = hand_pred_left
        return grad, batch["image"], extras

    def apply_sd(
        self,
        batch,
        weight=1,
        t=None,
        noise=None,
        w_schdl="dream",
        debug=False,
        text=None,
        grad_mode=None,
        **kwargs,
    ):
        self.model.zero_grad()  # should be unneccessary
        latents = batch["image"]
        text = self.text_template(text)
        if "text" in batch:
            batch["text"] = self.text_template(batch["text"])
        device = latents.device
        batch_size = len(latents)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        self.schedule_max_step(kwargs.get("it", 0), min_value=batch["image"].min())

        if t is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=device,
            )
        if not torch.is_tensor(t):
            t = torch.tensor(
                [t] * batch_size,
                dtype=torch.long,
                device=device,
            )
        if grad_mode is None:
            grad_mode = self.grad_mode
        if grad_mode == "noise":
            grad, latents, rtn = self._get_grad_in_noise_latents(
                batch, weight, t, noise, w_schdl, debug, text, **kwargs
            )
        elif grad_mode == "xstart_onestep":
            grad, latents, rtn = self._get_grad_in_xtart_onestep(
                batch, weight, t, noise, w_schdl, debug, text, **kwargs
            )
        else:
            raise NotImplementedError(f"Unknown grad_mode {self.grad_mode}")

        # grad = self.model.distribute_weight(grad, w_mask, w_normal, w_depth)  # interpolate your own image
        grad = torch.nan_to_num(grad)
        # latents.retain_grad()  # just for debug
        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # if not debug:
        if grad.ndim >= 1:
            # let's use the trick
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            batch_size = len(latents)
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            grad = loss_sds
            rtn["losses"] = {"diff_image": grad}
        rtn["t"] = t[0].item()
        rtn["latents"] = latents
        return grad, rtn

    def get_noisy_image(self, image, t, noise=None, encode=False):
        """return I_t"""
        # if there is encoder, encode image
        if hasattr(self.model, "encode") and encode:
            image = self.model.encode(
                image,
            )

        if noise is None:
            noise = torch.randn_like(image)
        if not torch.is_tensor(t):
            t = torch.tensor([t] * len(image), dtype=torch.long, device=image.device)
        noisy_image = self.diffusion.q_sample(image, t, noise)
        return noisy_image

    def get_cfg_model_fn(self, glide_model, guidance_scale, w2=1):
        """

        :param glide_model: _description_
        :param guidance_scale: _description_
        :return: a function that takes in x_t in shape of (N*2, C*2, H, W)
        but only use the first N batch of x_t, and return
        (2N, 2C, H, W),  where the :N, ... == N:2N, ...
        """
        th = torch
        w1 = guidance_scale

        def cfg_model_fn_1st(x_t, ts, **kwargs):
            # with classifier-free guidance
            _3 = x_t.shape[1] // 2
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)

            eps, rest = model_out[:, :_3], model_out[:, _3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)  # (N, C, H, W)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)  # 2*N, C, H, W
            return th.cat([eps, rest], dim=1)  # 2N, 2C, H, W

        def cfg_model_fn_2nd(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 3]
            combined = torch.cat(
                [
                    half,
                ]
                * 3,
                dim=0,
            )

            model_out = glide_model(combined, ts, **kwargs)
            C = model_out.shape[1]
            eps, rest = model_out[:, : C // 2], model_out[:, C // 2 :]

            cond_eps, first_eps, uncond_eps = torch.split(eps, len(eps) // 3, dim=0)
            half_eps = (
                uncond_eps + w1 * (first_eps - uncond_eps) + w2 * (cond_eps - first_eps)
            )

            eps = torch.cat([half_eps] * 3, dim=0)
            return torch.cat([eps, rest], dim=1)

        if self.cfg.get("cfg_order", 1) == 1:
            cfg_model_fn = cfg_model_fn_1st
        elif self.cfg.get("cfg_order", 1) == 2:
            cfg_model_fn = cfg_model_fn_2nd
        if self._warp is None:
            self._warp = self.diffusion._wrap_model(cfg_model_fn)
        return self._warp  # self.diffusion._wrap_model(cfg_model_fn)

    def get_model_kwargs(self, device, batch_size, cond_image, cond_text=None):
        """
        Args:
            device (_type_): _description_
            batch_size (_type_): _description_
            cond_image (_type_): dict
            cond_text (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # TODO: clean it
        uncond_image = self.model.cfg.get("uncond_image", False)
        assert cond_text is not None
        text = self.const_str = cond_text
        model_kwargs = self.model.get_model_kwargs(
            device, batch_size, cond_image, text, uncond_image
        )
        return model_kwargs

    def get_pred_noise(self, model_fn, latents_noisy, t, batch, cond_text=None):
        """
        inside the method, it
        :param model_fn: CFG func! Wrapped! function
        :param latents_noisy:in shape of (N, C, H, W)
        :param t: in shape of (N, )
        :return: (N, C, H, W)
        """
        # with CFG
        if not isinstance(model_fn, _WrappedModel):
            print("##### Should not appear, sd.py:L329")
            model_fn = self.diffusion._wrap_model(model_fn)
        batch_size = len(latents_noisy)
        device = latents_noisy.device

        with torch.no_grad():
            nb = self.cfg.get("cfg_order", 1) + 1
            latent_model_input = torch.cat([latents_noisy] * nb)
            # apply CF-guidance
            model_kwargs = self.get_model_kwargs(device, batch_size, batch, cond_text)
            # from # GaussinDiffusion:L633 get_eps()
            tt = torch.cat([t] * nb, 0)
            model_output = model_fn(latent_model_input, tt, **model_kwargs)

            # model_output = self.unet(latent_model_input, tt, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, _ = model_output
            noise_pred = model_output[:, : model_output.shape[1] // 2]
        return noise_pred[: len(noise_pred) // nb]
