from hydra import main
from glob import glob
import yaml
import os
import os.path as osp

result_dir = '/home/yufeiy2/scratch/result/vhoi/copy_load'
target_file = '/home/yufeiy2/scratch/result/vhoi/dev_minimal/Knife_1/config.yaml'


def batch_prune():
    config_list = glob(osp.join(result_dir, '*/config.yaml'))
    for config_file in config_list:
        cfg_src = yaml.load(open(config_file), Loader=yaml.FullLoader)
        cfg_tgt = yaml.load(open(target_file), Loader=yaml.FullLoader)
        
        missing_keys, unexpected_keys = find_unmatched_keys(cfg_src, cfg_tgt)
        print('missing keys: ', missing_keys)
        print('unexpected keys: ', unexpected_keys)
        print(len(missing_keys), len(unexpected_keys))

        skip_keys = ['engine', 'environment.wandb', 'environment.submitit_dir']
        unexpected_keys += skip_keys
        # remove unexpected keys from cfg_src
        cfg_src_copy = {}
        copy_dict_but(cfg_src, cfg_src_copy, unexpected_keys)

        print('------')
        missing_keys, unexpected_keys = find_unmatched_keys(cfg_src_copy, cfg_tgt)
        print('missing keys: ', missing_keys)
        print('unexpected keys: ', unexpected_keys)

        dst_file = osp.join(osp.dirname(config_file), 'config_pruned.yaml')
        with open(dst_file, 'w') as f:
            yaml.dump(cfg_src_copy, f)

        print(config_file)
        break
    return 


def copy_dict_but(src, tgt, keys, pref=''):
    for key in src.keys():
        if f'{pref}{key}' not in keys:
            if isinstance(src[key], dict):
                tgt[key] = {}
                copy_dict_but(src[key], tgt[key], keys, pref=f'{pref}{key}.')
            else:
                tgt[key] = src[key]
        else:
            # print(f'remove {pref}{key}')
            pass


def find_unmatched_keys(cfg_src, cfg_tgt):
    missing_keys = []
    unexpected_keys = []
    for key in cfg_tgt.keys():
        if key not in cfg_src.keys():
            missing_keys.append(key)
        else:
            if isinstance(cfg_tgt[key], dict):
                new_miss , new_unexp = find_unmatched_keys(cfg_src[key], cfg_tgt[key])
                missing_keys += [f'{key}.{e}' for e in new_miss] 
                unexpected_keys += [f'{key}.{e}' for e in new_unexp]
            if isinstance(cfg_tgt[key], dict) and not isinstance(cfg_src[key], dict):
                missing_keys.append(key)

    # find unexpected keys in cfg_src
    for key in cfg_src.keys():
        if key == 'logging':
            print('logging', cfg_src[key], cfg_tgt[key])
            # import pdb; pdb.set_trace()
        if key not in cfg_tgt.keys():
            unexpected_keys.append(key)
        else:
            if isinstance(cfg_tgt[key], dict):
                new_miss, new_unexp = find_unmatched_keys(cfg_src[key], cfg_tgt[key])
                missing_keys += [f'{key}.{e}' for e in new_miss]
                unexpected_keys += [f'{key}.{e}' for e in new_unexp]
            if isinstance(cfg_src[key], dict) and not isinstance(cfg_tgt[key], dict):
                new_miss, new_unexp = find_unmatched_keys(cfg_src[key], {})
                missing_keys += [f'{key}.{e}' for e in new_miss]
                unexpected_keys += [f'{key}.{e}' for e in new_unexp]
    return missing_keys, unexpected_keys


def batch_change_path():
    config_list = glob(osp.join(result_dir, '*/config_prune.yaml'))
    for config_file in config_list:
        cfg_src = yaml.load(open(config_file), Loader=yaml.FullLoader)
        replace(cfg_src, 'environment.data_dir', )
        replace(cfg_src, 'output', ['environment.output'])
        dst_file = osp.join(osp.dirname(config_file), 'config_path.yaml')
        with open(dst_file, 'w') as f:
            yaml.dump(cfg_src, f)
        print(config_file)
    return


def replace(cfg, key, skip_keys, ):
    matched_value = eval(f'cfg.{key}')
    # if isinstance(matched_value, dict):
    #     _replace(cfg, key, skip_keys)
    # else:
    #     print(f'replace {key}')
    #     cfg[key] = cfg[key]

    def _replace(cfg, key, skip_keys, pref=''):
        for k in cfg.keys():
            if isinstance(cfg[k], dict):
                _replace(cfg[k], key, skip_keys, pref=f'{pref}{k}.')
            else:
                if f'{pref}{k}' == key:
                    if k not in skip_keys:
                        print(f'replace {pref}{k}')
                        cfg[k] = cfg[key]
                    else:
                        print(f'skip {pref}{k}')
    return 


from jutils import slurm_utils
@main(config_path="../configs", config_name="diffhoi3d")
@slurm_utils.slurm_engine()
def save_config(args):
    from utils import io_util
    io_util.save_config(args, osp.join(args.exp_dir, 'config.yaml'))
    return 


def copy_weight():
    src_dir = '/home/yufeiy2/scratch/result/vhoi/monosdf_thick400k_mono0/{}_mono0_diff5000_joint'
    dst_dir = '/home/yufeiy2/scratch/result/vhoi/hoi4d/{}'
    cat_list = 'Mug,Bottle,Kettle,Bowl,Knife,ToyCar'.split(',')
    ind = [1, 2]
    index_list = [f'{i}_{j}' for i in cat_list for j in ind]
    for index in index_list:
        src_folder = src_dir.format(index)
        dst_folder = dst_dir.format(index)
        cmd = 'cp -r {}/ckpts {}/ckpts'.format(src_folder, dst_folder)
        print(cmd)
        os.system(cmd)


@main(config_path="../ddpm3d/configs", config_name="train_3dprior")
def save_prior_config(args):
    from omegaconf import OmegaConf
    datadict = OmegaConf.to_container(args, resolve=False)
    
    path = osp.join(args.exp_dir, 'config.yaml')
    with open(path, "w") as outfile:
        outfile.write("%s" % OmegaConf.to_yaml(datadict))

    return 

if __name__ == '__main__':
    # batch_prune()
    # batch_change_path()
    save_config()
    # copy_weight()

    # save_prior_config()