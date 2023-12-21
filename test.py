import hydra.utils as hydra_utils
import os
import os.path as osp 
from omegaconf import DictConfig, OmegaConf
import yaml
from hydra import main


def save_yaml(datadict, path):
    datadict = OmegaConf.to_container(datadict, resolve=False)
    with open(path, "w") as outfile:
        outfile.write("%s" % OmegaConf.to_yaml(datadict))
    return 

def load_yaml(config_path):
    config = OmegaConf.load(config_path)
    # with open(config_path, encoding='utf8') as yaml_file:
        
    #     config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    #     config = ForceKeyErrorDict(**config_dict)
    return config

@main(config_path="configs", config_name="eval")
def test(args):
    print(OmegaConf.to_yaml(args))
    save_yaml(args, osp.join(hydra_utils.get_original_cwd(), 'test.yaml'))
    print(args.output)
    cfg = load_yaml(osp.join(hydra_utils.get_original_cwd(), 'test.yaml'))
    print(cfg.output)
    cfg.environment.output = 'heheheh'
    print(cfg.output)
    return 

if __name__ == '__main__':
    test()