import argparse
import os
import os.path as osp
from glob import glob
from pathlib import Path

import mesh_to_sdf
import numpy as np
import torch
import trimesh
from joblib import Parallel, delayed
from jutils import geom_utils, image_utils, mesh_utils
from tqdm import tqdm

# https://github.com/hjwdzh/Manifold
manifold = "/home/yufeiy2/Packages/Manifold/build/manifold"
device = "cuda:0"
save_dir = "/home/yufeiy2/scratch/result/vis_sdf/"


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


def vis_samples(inp_list, out_list, med_list=None, lim=1.5):
    for i in range(len(inp_list)):
        inp_file = inp_list[i]
        index = osp.basename(inp_file).split(".")[0] + f"_{i:04d}"
        oObj = mesh_utils.load_mesh(inp_file, device=device)
        oObj.textures = mesh_utils.pad_texture(oObj, "red")

        uSdf, uTo = load_sdf_grid(out_list[i] + ".npz", True, True, device=device)
        oTu = geom_utils.inverse_rt(mat=uTo, return_mat=True)
        oSdf, _ = mesh_utils.transform_sdf_grid(uSdf, oTu, lim=lim, N=64)

        oSdf_mesh = mesh_utils.batch_grid_to_meshes(oSdf, 1, half_size=lim)
        oSdf_mesh.textures = mesh_utils.pad_texture(oSdf_mesh, "white")

        oScene = mesh_utils.join_scene([oObj, oSdf_mesh])
        image_list = mesh_utils.render_geom_rot_v2(oScene)
        image_utils.save_gif(image_list, osp.join(save_dir, index))


def norm_to_unit_ball(mesh: trimesh.Trimesh, max_boundary=0.9):
    # center mesh and renorm boundary to [-max_boundary, max_boundary] and report transformation
    # 1. center mesh
    mean = np.mean(mesh.vertices, axis=0)
    mesh.vertices = mesh.vertices - mean
    # 2. renorm boundary to [-max_boundary, max_boundary]
    scale = max_boundary / np.max(np.abs(mesh.vertices))
    mesh.vertices = mesh.vertices * scale
    transformation = np.eye(4)
    transformation[:3, :3] = scale * np.eye(3)
    transformation[:3, 3] = -mean * scale
    return mesh, transformation


def convert_ply_to_obj(inp_mesh_file, out_mesh_file):
    # by trimesh
    mesh = trimesh.load(inp_mesh_file)
    mesh.export(out_mesh_file)
    return


def manifold_mesh(inp_mesh_file, out_mesh_file, N=2000):
    os.makedirs(osp.dirname(out_mesh_file), exist_ok=True)
    ext = inp_mesh_file.split(".")[-1]
    if ext == "ply":
        tmp_file = out_mesh_file.replace(".obj", "_tmp.obj")
        convert_ply_to_obj(inp_mesh_file, tmp_file)
        inp_mesh_file = tmp_file
    cmd = manifold + " " + inp_mesh_file + " " + out_mesh_file + f" {N}"
    print(cmd)
    os.system(cmd)
    if ext == "ply":
        os.system(f"rm {tmp_file}")
    return


def batch_sdf(
    inp_list, out_list, med_list=None, N=128, fit_to_unit_cube=True, skip=True
):
    for i, (inp_file, out_file) in tqdm(
        enumerate(zip(inp_list, out_list)), total=len(inp_list)
    ):
        lock_file = out_file + ".lock"
        if skip and osp.exists(out_file + ".npz"):
            print("skip", out_file)
            continue

        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if skip:
                print("lose lock", out_file)
                continue
        if med_list is not None:
            med_file = med_list[i] + ".obj"
            manifold_mesh(inp_file, med_file, 2000)
            inp_file = med_file

        sdf, transformation = get_sdf_grid(inp_file, N, fit_to_unit_cube)
        np.savez_compressed(out_file, sdf=sdf, transformation=transformation)

        # os.rmdir(lock_file)
        os.system(f"rm -r {lock_file}")


def get_sdf_grid(mesh_file, N=64, fit_to_unit_cube=False, **kwargs):
    mesh = trimesh.load(mesh_file, process=False)
    if fit_to_unit_cube:
        mesh, transformation = norm_to_unit_ball(mesh)
    else:
        transformation = np.eye(4)

    mesh = mesh_utils.as_mesh(mesh)
    if not mesh.is_watertight:
        print("mesh is not watertight")
        error_file = mesh_file.replace("/all/", "/bad/")
        os.makedirs(error_file, exist_ok=True)
        print("non watertight mesh saved to", mesh_file)

    d, h, w = np.meshgrid(
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N),
    )

    xyz = np.stack([h, d, w], axis=-1)  # (N, N, N, 3)  # I'm so confused??
    xyz = xyz.reshape(-1, 3)  # (N**3, 3)

    sdf = mesh_to_sdf.mesh_to_sdf(mesh=mesh, query_points=xyz, **kwargs)
    sdf = sdf.reshape(N, N, N)

    return sdf, transformation


def get_hoi4d():
    inp_dir = "/data/sriram/hoi4d/HOI4D_CAD_Model_for_release/rigid"
    med_dir = "/data/sriram/hoi4d/mesh_sdf/manifold/hoi4d/all_2k/"
    save_dir = "mesh_sdf/SdfGrids/hoi4d/all_2k/"
    inp_list = glob(osp.join(inp_dir, "*/*.obj"))[:10]
    out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    # Path to str
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    # med_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    # med_list = [osp.join(med_dir, str(out_file)) for out_file in med_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, None

def get_arctic_overfit():
    ROOT_DIR = "/data/sriram/arctic/"
    save_dir = "data/arctic_proc"
    os.makedirs(save_dir, exist_ok=True)

    seq_name  = f"{ROOT_DIR}/outputs/processed_verts/seqs/s05/box_use_01.npy"
    seq = np.load(seq_name, allow_pickle=True).item()

    obj_name = "box"
    obj_verts = seq["world_coord"]["verts.object"]

    idx = 150 # random idx ...
    object = trimesh.load(f"{ROOT_DIR}/data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj", process=False)
    # Position object according to the processed vertices
    object.vertices = obj_verts[idx]

    object.export(f"{save_dir}/overfit.obj")

    inp_list = [f"{save_dir}/overfit.obj"]
    out_list = [f"{save_dir}/overfit_sdf.npz"]
    return inp_list, out_list, None

def get_oakink():
    data_dir = "/home/yufeiy2/scratch/data/OakInk"
    inp_dir = osp.join(data_dir, "meshes/manifold_2k/")
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/oakink/all_2k/")
    # med_dir = osp.join(data_dir, 'mesh_sdf/manifold/oakink/all_2k/')
    inp_list = sorted(glob(osp.join(inp_dir, "*.obj")))  # 'a40108 s12200 *.obj')))
    out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, None


def get_contactpose():
    data_dir = "/home/yufeiy2/scratch/data/ContactPose"
    inp_dir = osp.join(data_dir, "mesh_sdf/obj_mesh/contactpose/all/")
    med_dir = osp.join(data_dir, "mesh_sdf/manifold/contactpose/all/")
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/contactpose/all/")
    inp_list = sorted(glob(osp.join(inp_dir, "*.obj")))
    out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    med_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    med_list = [osp.join(med_dir, str(out_file)) for out_file in med_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, med_list


def get_mow():
    data_dir = "/home/yufeiy2/scratch/data/MOW"
    inp_dir = osp.join(data_dir, "models_norm")
    med_dir = osp.join(data_dir, "mesh_sdf/manifold/")
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/")
    inp_list = sorted(glob(osp.join(inp_dir, "*.obj")))
    out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    med_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    med_list = [osp.join(med_dir, str(out_file)) for out_file in med_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, med_list


def get_obman(split="test"):
    data_dir = "/home/yufeiy2/scratch/data/ObMan"
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/")
    med_dir = osp.join(data_dir, "mesh_sdf/manifold/")

    split_file = osp.join(data_dir, f"obman/{split}_cad_set.txt")
    cad_list = [line.strip() for line in open(split_file)]

    shape_dir = "/home/yufeiy2/scratch/data/ShapeNetCore"
    cad_dir = os.path.join(shape_dir, "{}", "models", "model_normalized.obj")

    inp_list, med_list, out_list = [], [], []
    for cad_index in cad_list:
        inp_list.append(cad_dir.format(cad_index))
        med_list.append(osp.join(med_dir, cad_index))
        out_list.append(osp.join(save_dir, cad_index))
    print(inp_list[0], out_list[0], med_list[0])
    return inp_list, out_list, med_list


def get_ycba():
    data_dir = "/home/yufeiy2/scratch/data/YCBAfford"
    inp_list = sorted(
        glob(osp.join(data_dir, "models/*/google_16k/model_watertight_2000def.obj"))
    )
    # med_dir = osp.join(data_dir, 'mesh_sdf/manifold/')
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/")
    # inp_list = sorted(glob(osp.join(inp_dir,  '*.ply')))
    out_list = [e.split("/")[-3] for e in inp_list]
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, None


def get_grab():
    data_dir = "/home/yufeiy2/scratch/data/GRAB"
    inp_dir = osp.join(data_dir, "tools/object_meshes/contact_meshes/")
    med_dir = osp.join(data_dir, "mesh_sdf/manifold/")
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/")
    inp_list = sorted(glob(osp.join(inp_dir, "*.ply")))
    out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    out_list = [osp.join(save_dir, str(out_file)) for out_file in out_list]
    med_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    med_list = [osp.join(med_dir, str(out_file)) for out_file in med_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, med_list


def get_dexycb():
    data_dir = "/home/yufeiy2/scratch/data/DexYCB"
    print(osp.join(data_dir, "models/*/textured_simple.obj"))
    inp_list = sorted(glob(osp.join(data_dir, "models/*/textured_simple.obj")))
    med_dir = osp.join(data_dir, "mesh_sdf/manifold/")
    save_dir = osp.join(data_dir, "mesh_sdf/SdfGrids/")
    # out_list = [Path(inp_file[:-4]).relative_to(inp_dir) for inp_file in inp_list]
    out_list = [osp.join(save_dir, e.split("/")[-2]) for e in inp_list]
    med_list = [osp.join(med_dir, e.split("/")[-2]) for e in inp_list]
    print(inp_list[0], out_list[0])
    return inp_list, out_list, med_list


def main():
    # list of (inp_file, out_file)
    med_list = None
    if args.ds == "hoi4d":
        inp_list, out_list, med_list = get_hoi4d()
    elif args.ds == "oakink":
        inp_list, out_list, med_list = get_oakink()
    elif args.ds == "contactpose":
        inp_list, out_list, med_list = get_contactpose()
    elif args.ds == "grab":
        inp_list, out_list, med_list = get_grab()
    elif args.ds == "ycba":
        inp_list, out_list, med_list = get_ycba()
    elif args.ds == "dexycb":
        inp_list, out_list, med_list = get_dexycb()
    elif args.ds == "mow":
        inp_list, out_list, med_list = get_mow()
    elif args.ds == "obman":
        inp_list, out_list, med_list = get_obman()
    elif args.ds == "arctic_overfit":
        inp_list, out_list, med_list = get_arctic_overfit()
    elif args.ds == "arctic":
        raise NotImplementedError
    if args.vis:
        vis_samples(inp_list[0:8], out_list[0:8], med_list)
        return

    K = args.K
    if K > 0:
        # parallel by K
        Parallel(n_jobs=K)(
            delayed(batch_sdf)(
                inp_list[i::K],
                out_list[i::K],
                med_list[i::K],
                N=128,
                fit_to_unit_cube=True,
                skip=args.skip,
            )
            for i in range(K)
        )
    else:
        batch_sdf(
            inp_list, out_list, med_list, N=64, fit_to_unit_cube=True, skip=args.skip
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--ds", default="oakink")
    parser.add_argument("--K", default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()
