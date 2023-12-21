import numpy as np
import os
import os.path as osp
from preprocess.make_sdf_grid import get_sdf_grid

# folder/
#    object_0/
#        oObj.obj  // object mesh
#        uSdf.npz  // sdf grid
#        obj.txt   // class name

def make_ho3d():
    # use dexYCB's npz
    src_dir = "/private/home/yufeiy2/scratch/data/DexYCB/"
    dst_dir = "/private/home/yufeiy2/scratch/result/GenGrasps/HO3D/"
    index_list = "003_cracker_box,006_mustard_bottle,011_banana,021_bleach_cleanser,035_power_drill,004_sugar_box,010_potted_meat_can,019_pitcher_base,025_mug,037_scissors".split(
        ","
    )
    for index in index_list:
        obj_file = osp.join(src_dir, "models", index, "textured_simple.obj")
        dst_file = osp.join(dst_dir, index, "oObj.obj")
        os.makedirs(osp.dirname(dst_file), exist_ok=True)
        os.system(f"cp {obj_file} {dst_file}")
        print(f"copy {obj_file} to {dst_file}")

        # get sdf grid
        sdf_file = osp.join(dst_dir, index, "uSdf.npz")
        sdf, transformation = get_sdf_grid(dst_file, 64, True)
        np.savez_compressed(sdf_file, sdf=sdf, transformation=transformation)

        # get text
        text = index
        with open(osp.join(dst_dir, index, "obj.txt"), "w") as f:
            f.write(text)
    return


if __name__ == "__main__":
    make_ho3d()