# reference: glide_text2im.text2im_mode.py:Text2ImUNet
# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

# TODO: add LDM style condition wrapper?
import torch
from copy import deepcopy
from utils.io_util import get_obj_from_str
from .openai_model_3d import UNet3DModel


class Text2GFUNet(UNet3DModel):
    def __init__(
        self,
        cfg,
    ):
        # init cond model
        text_cfg = cfg.text if "text" in cfg else cfg.model.text
        embedder = get_obj_from_str(text_cfg.target)
        text_cond_model = embedder(**text_cfg.params)
        self.text_embed_dim = text_cond_model.ndim
        unet_params = deepcopy(cfg.model.unet.params)
        legacy_unet = cfg.get("legacy_unet", True)
        unet_params["context_dim"] = self.text_embed_dim
        super().__init__(**unet_params, legacy_unet=legacy_unet)

        self.text_cond_model = text_cond_model

    def get_text_cond(self, text):
        if text is None:
            return None
        if isinstance(text, str):
            return self.text_cond_model(text)
        if isinstance(text[0], str):
            return self.text_cond_model(text)
        return text

    def forward(self, x, timesteps, nXyz=None, text=None, hA=None, **kwargs):
        """
        :param x: x_t? noisy SDF in shape of (N, C, D, H, W)
        :param timesteps: _description_
        :param tokens: _description_, defaults to None
        :param mask: _description_, defaults to None
        :param hA: _description_, defaults to None
        :return: _description_
        """
        text_cond = self.get_text_cond(text)
        # crossattn text
        cc = text_cond

        out = super().forward(x, timesteps, context=cc)
        # legacy issue
        out = torch.cat([out, out], dim=1)
        return out
