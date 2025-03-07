# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import copy
import glob
import importlib
import logging
import os
import shutil

import addict
import imageio
import numpy as np
import skimage
import torch
import yaml
from jutils import model_utils
from omegaconf import DictConfig, OmegaConf
from skimage.transform import rescale

from utils.print_fn import log


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_from_checkpoint(ckpt, cfg_file=None):
    if cfg_file is None:
        cfg_file = ckpt.split("checkpoints")[0] + "/config.yaml"
    cfg = OmegaConf.load(cfg_file)
    cfg.model.resume_ckpt = None  # save time to load base model :p
    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(
        cfg,
    )
    if hasattr(model, "init_model"):
        model.init_model()

    print("loading from checkpoint", ckpt)
    weights = torch.load(ckpt)["state_dict"]
    model_utils.load_my_state_dict(model, weights)
    return model


# --------------------------------------------------------
#  load data
# --------------------------------------------------------
def load_sdf_grid(sdf_file, tensor=False, batched=False, device="cpu"):
    """

    :return: (N, N, N), (4, 4) of numpy
    """
    obj = np.load(sdf_file)
    sdf = obj["sdf"]
    transformation = obj["transformation"]
    if tensor:
        sdf = torch.FloatTensor(sdf).to(device)[None]
        transformation = torch.FloatTensor(transformation).to(device)
    if batched:
        sdf = sdf[None]
        transformation = transformation[None]
    return sdf, transformation


# --------------------------------------------------------
#  IO for reconstruction
# --------------------------------------------------------
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def backup(backup_dir):
    """automatic backup codes"""
    log.info("=> Backing up... ")
    special_files_to_copy = []
    filetypes_to_copy = [".py"]
    subdirs_to_copy = ["", "dataio/", "models/", "tools/", "utils/"]

    # to support hydra
    if "PYTHONPATH" in os.environ:
        this_dir = os.environ["PYTHONPATH"]
    else:
        this_dir = "./"  # TODO
    cond_mkdir(backup_dir)
    # special files
    [
        cond_mkdir(os.path.join(backup_dir, os.path.split(file)[0]))
        for file in special_files_to_copy
    ]
    [
        shutil.copyfile(os.path.join(this_dir, file), os.path.join(backup_dir, file))
        for file in special_files_to_copy
    ]
    # dirs
    for subdir in subdirs_to_copy:
        cond_mkdir(os.path.join(backup_dir, subdir))
        files = os.listdir(os.path.join(this_dir, subdir))
        files = [
            file
            for file in files
            if os.path.isfile(os.path.join(this_dir, subdir, file))
            and file[file.rfind(".") :] in filetypes_to_copy
        ]
        [
            shutil.copyfile(
                os.path.join(this_dir, subdir, file),
                os.path.join(backup_dir, subdir, file),
            )
            for file in files
        ]
    log.info("done.")



#-----------------------------
# configs
#-----------------------------

def save_config(datadict, path: str):
    datadict = copy.deepcopy(datadict)
    datadict.training.ckpt_file = None

    if isinstance(datadict, DictConfig):
        logging.warning("to yaml ")
        datadict = OmegaConf.to_container(datadict, resolve=False)
        with open(path, "w") as outfile:
            outfile.write("%s" % OmegaConf.to_yaml(datadict))
    else:
        logging.warning("to dict ")
        log.info(type(datadict))
        with open(path, "w", encoding="utf8") as outfile:
            yaml.dump(datadict.to_dict(), outfile, default_flow_style=False)


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1.0 / downscale, anti_aliasing=False, channel_axis=-1)

    # NOTE: pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    img = img.transpose(2, 0, 1)
    if img.shape[0] == 4:
        img = img[:3]
    return img


def load_mask(path, downscale=1):
    alpha = imageio.imread(path, mode='F')
    alpha = skimage.img_as_float32(alpha)
    if downscale != 1:
        alpha = rescale(alpha, 1.0 / downscale, anti_aliasing=False)
    object_mask = alpha > 127.5

    return object_mask
