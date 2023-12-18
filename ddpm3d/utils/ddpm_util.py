## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import numpy as np
import torch as th
from glide_text2im.model_creation import (
    get_named_beta_schedule,
)
from glide_text2im.respace import SpacedDiffusion, space_timesteps


def create_gaussian_diffusion(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = make_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )


def make_beta_schedule(schedule, n_timestep, linear_start=0.0015, linear_end=0.0195, cosine_s=8e-3):
    """LDM"""
    if schedule == "linear":
        betas = (
                th.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=th.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                th.arange(n_timestep + 1, dtype=th.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = th.linspace(linear_start, linear_end, n_timestep, dtype=th.float64)
    elif schedule == "sqrt":
        betas = th.linspace(linear_start, linear_end, n_timestep, dtype=th.float64) ** 0.5
    else:
        try:
            betas = get_named_beta_schedule(schedule, n_timestep)
            betas = th.FloatTensor(betas)
        except NotImplementedError:
            pass
    return betas.numpy()

