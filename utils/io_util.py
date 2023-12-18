# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import numpy as np
import importlib
import torch
from omegaconf import OmegaConf
from jutils import model_utils


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_from_checkpoint(ckpt, cfg_file=None):
    if cfg_file is None:
        cfg_file = ckpt.split('checkpoints')[0] + '/config.yaml'
    cfg = OmegaConf.load(cfg_file)
    cfg.model.resume_ckpt = None  # save time to load base model :p
    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(cfg, )
    if hasattr(model, 'init_model'):
        model.init_model()

    print('loading from checkpoint', ckpt)    
    weights = torch.load(ckpt)['state_dict']
    model_utils.load_my_state_dict(model, weights)
    return model

# --------------------------------------------------------
#  load data 
# --------------------------------------------------------
def load_sdf_grid(sdf_file, tensor=False, batched=False, device='cpu'):
    """

    :return: (N, N, N), (4, 4) of numpy
    """
    obj = np.load(sdf_file)
    sdf = obj['sdf']
    transformation = obj['transformation']
    if tensor:
        sdf = torch.FloatTensor(sdf).to(device)[None]
        transformation = torch.FloatTensor(transformation).to(device)
    if batched:
        sdf = sdf[None]
        transformation = transformation[None]
    return sdf, transformation
