# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import importlib
import wandb
import logging

import os
import os.path as osp
from glide_text2im.respace import SpacedDiffusion
from hydra import main

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from ddpm3d.dataset.dataset import build_dataloader
from ddpm3d.models.text2sdf import Text2GFUNet
from ddpm3d.utils.logger import LoggerCallback, build_logger
from jutils import model_utils, mesh_utils, image_utils, hand_utils, slurm_utils
from .utils.ddpm_util import create_gaussian_diffusion


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%m-%d %H:%M:%S",
)
torch.backends.cudnn.benchmark = True

def freeze_all_except(model, trainable_layers):
    for name, param in model.named_parameters():
        if name in trainable_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

class BaseModule(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.enable_bimanual = cfg.get("enable_bimanual", False)
        self.template_size = None
        # self.glide_model, self.diffusion, self.glide_options = build_glide_model(cfg)
        self.glide_model: Text2GFUNet = None
        self.diffusion: SpacedDiffusion = None
        self.glide_options: dict = {}

        self.val_batch = None
        self.train_batch = None
        self.log_dir = osp.join(cfg.exp_dir, "log")

        self.hand_wrapper = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean)
        if self.enable_bimanual:
            self.hand_wrapper_left = hand_utils.ManopthWrapper(cfg.environment.mano_dir, flat_hand_mean=cfg.flat_hand_mean, side="left")

    def train_dataloader(self):
        cfg = self.cfg
        dataloader = build_dataloader(
            cfg,
            cfg.trainsets,
            None,
            self.cfg.model.text_ctx,
            True,
            cfg.batch_size,
            True,
            workers=8,
        )
        for data in dataloader:
            self.train_batch = data
            break
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def val_dataloader(self):
        cfg = self.cfg
        val_dataloader = build_dataloader(
            cfg,
            cfg.testsets,  # cfg.data.test_data_dir,
            None,
            self.cfg.model.text_ctx,
            # cfg.data.test_split,
            is_train=False,
            bs=cfg.test_batch_size,
            shuffle=True,
            workers=8,
        )
        for data in val_dataloader:
            self.val_batch = data
            break
        return val_dataloader

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [x for x in self.glide_model.parameters() if x.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.adam_weight_decay,
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        self.set_inputs(batch, batch_idx)
        os.makedirs(self.log_dir, exist_ok=True)
        self.vis_input(batch, f"{batch_idx}_input", {})

        samples, sample_list = self(batch, guidance_scale=4)
        self.vis_samples(batch, samples, [], "%d_" % batch_idx, {}, self.global_step)

        N = len(batch["offset"])
        batch["offset"] = 1 / 64 + torch.zeros_like(batch["offset"])
        batch["nXyz"] = batch["offset"].reshape(N, 1, 1, 1, 3) + batch["nXyz"]
        samples, sample_list = self(batch, guidance_scale=4)
        self.vis_samples(
            batch, samples, [], "%d_off_" % batch_idx, {}, self.global_step
        )

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch, batch_idx)
        self.train_batch = batch
        loss, losses = self.step(batch, batch_idx)

        if self.global_step % self.cfg.print_frequency == 0:
            losses["loss"] = loss
            self.logger.log_metrics(losses, step=self.global_step)
            logging.info("[%05d]: %f" % (self.global_step, loss))
            for k, v in losses.items():
                logging.info("\t %08s: %f" % (k, v.item()))

        if self.global_step % self.cfg.log_frequency == 0:
            log = {}
            max_bs = min(8, len(batch["image"]))
            smaller_batch = {k: v[:max_bs] for k, v in batch.items()}
            with torch.no_grad():
                self.vis_input(smaller_batch, "train_input/", log)
                self.generate_sample_step(smaller_batch, "train/", log, S=1)
                self.logger.log_metrics(log, step=self.global_step)
                print("\t %08s: %f" % (k, v.item()))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        log = {}
        self.set_inputs(batch, batch_idx)
        val_batch = model_utils.to_cuda(batch, batch["hA"])
        self.vis_input(batch, "val_input/batch", log)
        self.generate_sample_step(val_batch, "val/batch", log, S=1)

        self.logger.log_metrics(log, self.global_step)

    def encode(self, x):
        return x

    def decode(self, z):
        return {"image": z}

    @torch.no_grad()
    def forward(self, batch, second_order=False, **kwargs):
        if second_order:
            return self.second_order_forward(batch, **kwargs)
        cfg = self.cfg
        guidance_scale = kwargs.pop("guidance_scale", cfg.test_guidance_scale)
        sample_respacing = kwargs.pop("sample_respacing", cfg.sample_respacing)
        uncond_image = kwargs.pop("uncond_image", cfg.get("uncond_image", False))
        device = batch["image"].device
        N = len(batch["image"])
        model_kwargs = self.get_model_kwargs(
            device,
            N,
            batch,
            batch["text"],
        )
        samples, sample_list = self.sample(
            guidance_scale=guidance_scale,
            device=self.device,
            prediction_respacing=sample_respacing,
            batch_size=N,
            uncond_image=uncond_image,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        samples = self.decode(samples)
        return samples, sample_list

    def sample(
        self,
        val_batch=None,
        guidance_scale=4,
        device="cpu",
        prediction_respacing="100",
        model_kwargs=None,
        batch_size=None,
        hijack={},  # {'x0': latents, 'mask': mask,}
        **kwargs,
    ):
        glide_model = self.glide_model
        glide_options = self.glide_options
        size = self.template_size
        eval_diffusion = create_gaussian_diffusion(
            steps=glide_options["diffusion_steps"],
            noise_schedule=glide_options["noise_schedule"],
            timestep_respacing=prediction_respacing,
        )
        if batch_size is None:
            # Create the text tokens to feed to the model.
            batch_size = len(val_batch["hA"])
            # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2

        if model_kwargs is None:
            # val_batch = model_utils.to_device(val_batch, device)
            uncond = glide_model.get_text_cond([""] * batch_size)
            cond = glide_model.get_text_cond(val_batch["text"])
            text = torch.cat([cond, uncond], dim=0)

            hA = torch.cat([val_batch["hA"], torch.zeros_like(val_batch["hA"])], dim=0)
            nXyz = torch.cat([val_batch["nXyz"]] * 2, dim=0)

            # Pack the tokens together into model kwargs.
            model_kwargs = dict(
                text=text,
                hA=hA,
                nXyz=nXyz,
            )

        def cfg_model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)
            C = model_out.shape[1]
            eps, rest = model_out[:, : C // 2], model_out[:, C // 2 :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        model_fn = cfg_model_fn
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            [
                full_batch_size,
            ]
            + size,  
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            # hijack=hijack,
        )[:batch_size]
        return samples, []

    @torch.no_grad()
    def generate_sample_step(
        self, batch, pref, log, step=None, S=2, guidance_scale=None
    ):
        cfg = self.cfg
        if step is None:
            step = self.global_step
        file_list = []
        step = self.global_step
        if guidance_scale is None:
            guidance_scale = cfg.test_guidance_scale
        for n in range(S):
            samples, sample_list = self(batch)
            self.vis_samples(batch, samples, [], pref + "%d_" % n, log, step)
        return file_list, samples

    def decode_samples(self, sample, offset):
        """Extract Object surface~~
        :param tensor: (N, 1, D, H, W)
        :return: Meshes
        """
        tensor = sample["image"]
        N = len(tensor)
        jObj = mesh_utils.batch_grid_to_meshes(
            tensor.squeeze(1), N, half_size=self.cfg.side_lim, offset=offset
        )
        jObj.textures = mesh_utils.pad_texture(jObj)

        return {
            "jObj": jObj,
        }

    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):
        return

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        if step is None:
            step = self.global_step
        jObj = batch['image']  # emmm welll we should denorm it.... 
        N = len(jObj)
        jObj = mesh_utils.batch_grid_to_meshes(jObj.squeeze(1), N, half_size=self.cfg.side_lim)
        jObj.textures = mesh_utils.pad_texture(jObj)

        self.vis_meshes(jObj, f'{pref}_jObj', log, step)

        hHand,  _ = self.hand_wrapper(None, batch['hA'])
        nTh = hand_utils.get_nTh(hand_wrapper=self.hand_wrapper, hA=batch['hA'])
        jHand = mesh_utils.apply_transform(hHand, nTh)
        jHand.textures = mesh_utils.pad_texture(jHand, 'blue')
        jHoi = mesh_utils.join_scene([jHand, jObj])

        if self.enable_bimanual:
            hHand_left,  _ = self.hand_wrapper_left(None, batch['hA_left'])
            jHand_left = mesh_utils.apply_transform(hHand_left, batch["nTh_left"][:,0])
            jHand_left.textures = mesh_utils.pad_texture(jHand_left, 'blue')
            jHoi = mesh_utils.join_scene([jHoi, jHand_left])

        caption = " | ".join(batch['fname'])
        self.vis_meshes(jHoi, f'{pref}_sample_jHoi', log, step, caption=caption)

    def vis_meshes(self, meshes, name, log, step=None, text=None, caption=None):
        image_list = mesh_utils.render_geom_rot_v2(meshes)
        fname = osp.join(self.log_dir, f"{name}_{step}")
        image_utils.save_gif(image_list, fname, text_list=[text])
        log[f"{name}"] = wandb.Video(fname + ".gif", caption=caption)
        return log

    def get_model_kwargs(
        self,
        device,
        batch_size,
        val_batch,
        text="",
        uncond_image=False,
    ):
        glide_model = self.glide_model

        uncond = glide_model.get_text_cond([""] * batch_size).to(device)
        cond = glide_model.get_text_cond(text).to(device)
        text = torch.cat([cond, uncond], dim=0)

        hA = torch.cat([val_batch["hA"], torch.zeros_like(val_batch["hA"])], dim=0)
        # hA = torch.cat([val_batch['hA'], val_batch['hA']], dim=0)

        if "nXyz" not in val_batch:
            N = len(val_batch["hA"])
            reso = self.cfg.side_x
            lim = self.cfg.side_lim
            nXyz = mesh_utils.create_sdf_grid(N * 2, reso, lim, device=device)
        else:
            nXyz = torch.cat([val_batch["nXyz"]] * 2, dim=0)

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            text=text,
            hA=hA,
            nXyz=nXyz,
        )
        return model_kwargs


@main("configs", "train_3dprior_bimanual", version_base=None)
@slurm_utils.slurm_engine()
def main_worker(cfg):
    # handle learning rate
    print("main worker")

    os.makedirs(cfg.exp_dir, exist_ok=True)
    with open(
        osp.join(
            cfg.exp_dir,
            "config.yaml",
        ),
        "w",
    ) as fp:
        OmegaConf.save(cfg, fp, True)

    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(cfg)

    # initialize ae
    if osp.exists(cfg.ckpt):
        model = model_utils.load_from_checkpoint(cfg.ckpt, cfg=cfg)

    if cfg.model.freeze_transformer:
        model_utils.freeze(model.glide_model.text_cond_model)
    # model.cuda()

    if cfg.freeze_except_input_output:
        # Freeze all layers except the input/output layers for modified input/output space
        trainable_layers = ['glide_model.input_blocks.0.0.weight', 'glide_model.out.2.weight', 'glide_model.out.2.bias']
        model = freeze_all_except(model, trainable_layers)

    logger = build_logger(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="step",
        save_top_k=cfg.save_topk,
        mode="max",
        every_n_train_steps=cfg.save_frequency,
        save_last=True,
        dirpath=osp.join(cfg.exp_dir, "checkpoints"),
    )

    val_kwargs = {}
    n_iters = len(model.train_dataloader()) // torch.cuda.device_count()
    if n_iters < cfg.log_frequency:
        val_kwargs["check_val_every_n_epoch"] = int(cfg.log_frequency) // n_iters
    else:
        val_kwargs["val_check_interval"] = cfg.log_frequency
    model_summary = ModelSummary(2)
    callbacks = [model_summary, checkpoint_callback, LoggerCallback()]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.ngpu,
        strategy="ddp_find_unused_parameters_false",
        num_sanity_val_steps=cfg.sanity_step,
        limit_val_batches=1,
        default_root_dir=cfg.exp_dir,
        logger=logger,
        max_steps=cfg.max_steps,
        callbacks=callbacks,
        gradient_clip_val=cfg.model.grad_clip,
        **val_kwargs,
    )
    trainer.strategy.barrier()
    trainer.fit(model)

    return model


if __name__ == "__main__":
    main_worker()
