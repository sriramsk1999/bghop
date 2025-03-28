
import argparse
import os
import trimesh
import json
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

def quaternion_raw_multiply(a, b):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_invert(quaternion):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_apply(quaternion, point):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def load_object_data(root, obj_type):
    obj = trimesh.load(os.path.join(root, 'meta/object_vtemplates', obj_type, 'mesh.obj'), process=False)
    obj.vertices = obj.vertices / 1000. # ARCTIC code also scales down the object by 1000 ...
    obj_verts = torch.from_numpy(obj.vertices)

    with open(os.path.join(root, 'meta/object_vtemplates', obj_type, 'parts.json'), 'r') as f:
        parts = np.array(json.load(f), dtype=np.bool)
    obj_parts = torch.LongTensor(parts) + 1

    return obj, obj_verts, obj_parts

def articulate_top_half(obj, obj_verts, obj_parts, articulation_range):
    top_mask = obj_parts == 1
    obj_verts_top = obj_verts[top_mask]

    z_axis = torch.Tensor([[0., 0., -1.]])
    for arti in articulation_range:
        quat_arti = axis_angle_to_quaternion(z_axis * arti)
        obj_verts_top_articulated = quaternion_apply(quat_arti[None, :], obj_verts_top[None])

        new_obj = obj.copy()
        new_obj.vertices[top_mask] = obj_verts_top_articulated[0].numpy()

        yield arti, new_obj

def save_mesh(new_obj, output_dir, obj_type, articulation):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{obj_type}_{articulation:.2f}.obj')
    new_obj.export(output_path)

def process_objects(root, output_dir, articulation_range):
    objects = os.listdir(os.path.join(root,"meta/object_vtemplates"))

    for obj_type in tqdm(objects):
        obj, obj_verts, obj_parts = load_object_data(root, obj_type)

        for articulation, articulated_obj in articulate_top_half(obj, obj_verts, obj_parts, articulation_range):
            save_mesh(articulated_obj, output_dir, obj_type, articulation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and articulate object meshes before converting to SDF.")
    parser.add_argument("--root", type=str, default="/data/sriram/arctic/data/arctic_data/data/", help="Root directory for ARCTIC data")
    parser.add_argument("--start_angle", type=float, default=0, help="Start of articulation range in radians")
    parser.add_argument("--end_angle", type=float, default=1.5*np.pi, help="End of articulation range in radians")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of articulations to generate")

    args = parser.parse_args()
    articulation_range = np.linspace(args.start_angle, args.end_angle, args.n_steps)

    output_dir = f"{args.root}/arctic_mesh"
    process_objects(args.root, output_dir, articulation_range)
