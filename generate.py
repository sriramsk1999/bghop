# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

"""script to debug SDLoss for 3D prior"""
import json
from hydra import main
import hydra.utils as hydra_utils
import os
import os.path as osp
import torch
from jutils import model_utils, mesh_utils, hand_utils, slurm_utils, image_utils
from utils.obj2text import Obj2Text

device = "cuda:0"
oakink_list = [
    "mug",
    "bowl",
    "power_drill",
    "pen",
    "toothbrush",
    "wineglass",
    "fryingpan",
    "hammer",
    "knife",
    "cameras",
    "apple",
    "banana",
    "flashlight",
]


@main(config_path="ddpm3d/configs", config_name="vis_3dprior_bimanual", version_base=None)
@slurm_utils.slurm_engine()
def sample_hoi(args):
    torch.manual_seed(args.seed)
    model = model_utils.load_from_checkpoint(args.load_pt)
    enable_bimanual = model.cfg.get("enable_bimanual", False)

    model = model.to(device)
    lim = model.cfg.side_lim
    # change
    save_dir = args.vis_dir
    model.log_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    lib_name = model.cfg.get("lib", None)
    if lib_name is not None:
        lib_name = osp.join(hydra_utils.get_original_cwd(), f"docs/{lib_name}.json")
    text_template = Obj2Text(lib_name, enable_bimanual=enable_bimanual)

    def do_one(cat):
        S = args.S
        text = text_template(cat)
        print(f"Prompt - {text}")

        cat_sem = text.split("grasping a ")[-1].split(",")[0]
        pref = osp.join(save_dir, f"{cat_sem}")
        save_file = pref + f"_jHoi_{S-1:2d}.obj"
        lock_file = save_file + ".lock"
        if osp.exists(save_file):
            return
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            return

        reso = 16
        batch = {
            "image": torch.randn([S, 1, reso, reso, reso], device=device),
            "hA": torch.randn([S, 45], device=device),
            "nXyz": mesh_utils.create_sdf_grid(S, reso, lim, device=device),
            "text": [text] * S,
            "offset": torch.zeros(S, 3).to(device),
        }
        batch = model_utils.to_cuda(batch, device)
        samples, sample_list = model(batch)

        dec = model.decode_samples(samples, batch["offset"])
        jObj = dec["jObj"]
        jObj.textures = mesh_utils.pad_texture(jObj, "yellow")

        if "hand" in samples:
            hA, _, _, _ = model.hand_cond.grid2pose_sgd(
                samples["hand"], field=model.cfg.field
            )
        else:
            hA = samples["hA"]

        hHand, _ = model.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hand_wrapper=model.hand_wrapper, hA=hA)
        jHand = mesh_utils.apply_transform(
            hHand,
            nTh,
        )
        jHand.textures = mesh_utils.pad_texture(jHand, "blue")
        jHoi = mesh_utils.join_scene([jHand, jObj])

        if enable_bimanual:
            hA_left, _, _, nTh_left = model.hand_cond_left.grid2pose_sgd(
                samples["hand_left"], field=model.cfg.field, is_left=True
            )
            hHand_left, _ = model.hand_wrapper_left(None, hA_left)
            jHand_left = mesh_utils.apply_transform(
                hHand,
                nTh_left[:, 0],
            )
            jHand_left.textures = mesh_utils.pad_texture(jHand_left, "blue")
            jHoi = mesh_utils.join_scene([jHoi, jHand_left])

        mesh_utils.dump_meshes(pref + "_jHoi", jHoi)
        image_list = mesh_utils.render_geom_rot_v2(jHoi)
        image_utils.save_gif(image_list, pref + "_jHoi", max_size=2048)

        os.rmdir(lock_file)

    if lib_name is not None:
        lib = json.load(open(lib_name))
    else:
        lib = {}

    for cat in args.cat_list.split("+"):
        if cat not in lib:
            print(f"{cat} not in dictionary")
            print(f'check list of available objects in "{lib_name}"')
        # for cat in lib.keys():
        do_one(cat)


if __name__ == "__main__":
    sample_hoi()
