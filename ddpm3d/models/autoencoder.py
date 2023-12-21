import wandb
import os.path as osp
from typing import Any
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from ddpm3d.dataset.dataset import build_dataloader
from jutils import mesh_utils
from utils.hand_utils import build_hand_field
from ..vqvae.quantizer import VectorQuantizer
from ..vqvae.vqvae_modules import Encoder3D, Decoder3D
from utils.viz_utils import Visualizer


class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx=0,
        global_step=0,
        split="train",
        **kwargs,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_codebook": codebook_loss.detach().mean(),
            "loss_nll": nll_loss.detach().mean(),
            "loss_rec": rec_loss.detach().mean(),
        }

        return loss, log


class BaseAutoencoder(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.viz = Visualizer(cfg, self.log_dir)
        # self.hand_cond = build_hand_field(cfg.get("field", "none"), cfg)
        # self.hand_dim = hand_dim = self.hand_cond.ndim
        
        self.hand_cond = build_hand_field('none', cfg)
        self.hand_dim = hand_dim = self.hand_cond.ndim

        self.register_buffer("mean", torch.zeros([1, 1 + hand_dim, 1, 1, 1]))
        if self.cfg.get("norm_inp", False):
            nsdf_std = self.cfg.get("std", 0.08)
            nsdf_hand = self.cfg.get("std_hand", 0.18)
            std = torch.FloatTensor([nsdf_std] + [nsdf_hand] * hand_dim).reshape(
                1, -1, 1, 1, 1
            )
            self.register_buffer("std", std)
        else:
            self.register_buffer("std", torch.ones([1, 1 + hand_dim, 1, 1, 1]))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @property
    def log_dir(
        self,
    ):
        return osp.join(self.cfg.exp_dir, "log")

    def train_dataloader(self):
        cfg = self.cfg
        dataloader = build_dataloader(
            cfg, cfg.trainsets, None, None, True, cfg.batch_size, True
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def val_dataloader(self):
        cfg = self.cfg
        val_dataloader = build_dataloader(
            cfg,
            cfg.testsets,  # cfg.data.test_data_dir,
            None,
            None,
            # cfg.data.test_split,
            False,
            cfg.test_batch_size,
            True,
        )
        return val_dataloader

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, x):
        return x * self.std + self.mean

    def get_input(self, batch, k):
        self.set_inputs(batch)
        x = batch["image"]
        return x

    # @torch.no_grad()
    def set_inputs(self, batch, soft="hard"):
        if self.cfg.gpu_trans and "nSdf" not in batch:
            lim = self.cfg.side_lim
            H = self.cfg.side_x
            uSdf = batch["uSdf"]
            nTu = batch["nTu"]
            r = self.cfg.get("jitter_x", 0.0) if self.training else 0.0
            offset = torch.rand([len(uSdf), 3], device=uSdf.device) * r * 2 - r
            nSdf, nXyz = mesh_utils.transform_sdf_grid(
                uSdf, nTu, N=H, lim=lim, offset=offset
            )
            batch["nSdf"] = nSdf
            batch["nXyz"] = nXyz
            batch["offset"] = offset

        image = batch["nSdf"]
        if self.cfg.tsdf is not None:
            image = image.clamp(min=-self.cfg.tsdf, max=self.cfg.tsdf)
        if self.hand_cond.ndim > 0:
            hand = self.hand_cond(
                batch["hA"],
                image.shape[-1],
                batch["nXyz"],
                field=self.cfg.field,
                rtn_wrist=False,
            )
            if self.cfg.tsdf_hand is not None:
                hand = hand.clamp(min=-self.cfg.tsdf_hand, max=self.cfg.tsdf_hand)
            image = torch.cat([image, hand], dim=1)
        image = self.norm(image)
        batch["image"] = image
        return batch

    @rank_zero_only
    def vis_step(self, x, pref, log, step=None):
        """
        Args:
            x (_type_): (N, C, D ,H, W), C is either 1 or 1+48?
            pref (_type_): _description_
            log (_type_): _description_
            step (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        x = self.unnorm(x)

        jObj = x[:, 0:1, ...]
        N = len(jObj)
        lim = self.cfg.side_lim
        if self.cfg.get("rep", "sdf") == "occ":
            jObj = -jObj
        jObj = mesh_utils.batch_grid_to_meshes(jObj.squeeze(1), N, half_size=lim)
        jObj.textures = mesh_utils.pad_texture(jObj)
        self.viz.render_meshes(jObj, f"{pref}_jObj", log, step)
        nTw = mesh_utils.get_nTw(jObj)
        self.viz.render_grids(x[:, 0:1, ...], nTw, f"{pref}_jGridObj", log, step)

        # sgd_hands
        if x.shape[1] > 1:
            hA, hA_list, rtn = self.hand_cond.grid2pose_sgd(
                x[:, 1:], self.cfg.sgd_hand, field=self.cfg.field
            )
            self.viz.render_tsdf_hand(
                rtn["field"][0:1], hA[0:1], f"{pref}_tsdf_hand", log, step
            )
            self.viz.render_hA_traj(hA_list, f"{pref}_hA_traj", log, step, nTw.device)
            self.viz.render_hoi(jObj, hA, f"{pref}_jHoi", log, step)
            self.viz.render_hand(hA, f"{pref}_hHand", log, step)
        return log


class VQModel(BaseAutoencoder):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        batch_resize_range=None,
        scheduler_config=None,
        lr_g_factor=1.0,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        use_ema=False,
        cfg={},
    ):
        super().__init__(cfg)
        self.embed_dim = embed_dim
        ddconfig["in_channels"] += self.hand_dim
        ddconfig["out_ch"] += self.hand_dim
        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)
        self.loss = VQLoss(**lossconfig.params)  # instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(
                f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}."
            )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        # if optimizer_idx == 0:
        # autoencode
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            predicted_indices=ind,
        )

        log_dict_ae = {f"train/{k}": v for k, v in log_dict_ae.items()}
        self.logger.log_metrics(log_dict_ae, step=self.global_step)
        if self.global_step % 100 == 0:
            # pretty print log_dict_ae
            print(f"Step {self.global_step}:")
            for k, v in log_dict_ae.items():
                if isinstance(v, torch.Tensor):
                    print(f"\t{k}: {v.item():.4f}")
                else:
                    print(f"\t{k}: {v}")
        return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict, xrec = self._validation_step(batch, batch_idx)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        self.viz.render_hand(
            batch["hA"], "val_input/gt_hand", log_dict, self.global_step
        )
        print("input")
        self.vis_step(batch["image"], "val_input/", log_dict, self.global_step)
        self.vis_step(xrec, "val_recon/", log_dict, self.global_step)

        code = self.quantize.embedding.weight  # K, D
        for c in range(code.shape[-1]):
            log_dict[f"code/dim_{c}"] = wandb.Histogram(code[:, c].cpu())
        self.logger.log_metrics(log_dict, self.global_step)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + suffix,
            predicted_indices=ind,
        )

        return log_dict_ae, xrec

    def configure_optimizers(self):
        lr_g = self.lr_g_factor * self.cfg.learning_rate
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(self.parameters(), lr=lr_g, betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.StepLR(opt_ae, 1000, 0.9)

        return [
            opt_ae,
        ], [
            scheduler,
        ]

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VQModelWrapper(nn.Module):
    def __init__(self, model: VQModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = model.embed_dim
        self.downsample = model.downsample
        self.model = model

    def set_inputs(self, *args, **kwargs):
        return self.model.set_inputs(*args, **kwargs)

    def norm(self, x):
        return self.model.norm(x)

    def unnorm(self, x):
        return self.model.unnorm(x)

    def encode(self, x):
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.model.quantize(h)
        else:
            quant = h
        quant = self.model.post_quant_conv(quant)
        dec = self.model.decoder(quant)
        return dec


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
