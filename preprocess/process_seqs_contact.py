import argparse
import json
import sys
import time
import traceback
from glob import glob

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

sys.path = ["."] + sys.path
from common.body_models import construct_layers

# from src.arctic.models.object_tensors import ObjectTensors
from common.object_tensors import ObjectTensors
from src.arctic.processing import process_seq

from scipy.spatial import KDTree
from pathlib import Path
import pandas as pd

def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_verts", action="store_true")
    parser.add_argument("--mano_p", type=str, default=None)
    args = parser.parse_args()
    return args

def split_df_train_val(df, frac=0.9):
    fnames = df['filename'].unique()
    np.random.shuffle(fnames)
    split = int(frac * len(fnames))
    train_fnames = fnames[:split]
    val_fnames = fnames[split:]
    train_df = df[df['filename'].isin(train_fnames)]
    val_df = df[df['filename'].isin(val_fnames)]
    return train_df, val_df

def get_captions(filename, obj_type, sequence_length):
    """
    Reads a text file with timestamped captions and returns a list where each index
    corresponds to a timestep, containing the caption matching the keyword if applicable,
    or None otherwise.

    Args:
        filename (str): Path to the text file.
        sequence_length (int): Total length of the sequence (size of the output list).

    Returns:
        list: A list of length sequence_length with captions or None at each timestep.
    """
    with open(filename, 'r') as f:
            lines = f.readlines()

    captions = [None] * sequence_length
    for line in lines:
        # Remove leading/trailing whitespace (including newlines)
        line = line.strip()
        if not line: continue
        # Split into range and caption (e.g., "19-38" and "grasp, both hands")
        range_str, raw_caption = line.split(' ', 1)
        # Parse start and end from the range (e.g., "19-38" -> 19, 38)
        start, end = map(int, range_str.split('-'))

        if "both hands" in raw_caption:
            caption = raw_caption.replace(',',f' {obj_type} with')
            for t in range(start, end + 1):
                captions[t] = caption
    return captions

def load_description(desc_f, seq_len):
    """
    Descriptions come from text2hoi.
    """

def find_contact_timesteps(left_hand_sequence, right_hand_sequence, object_sequence, epsilon=0.01, fraction_threshold=0.2):
    """
    Find timesteps where both hands are in contact with the object.

    Parameters:
    - left_hand_sequence: np.array of shape (T, N_left, 3), left hand point clouds
    - right_hand_sequence: np.array of shape (T, N_right, 3), right hand point clouds
    - object_sequence: np.array of shape (T, N_obj, 3), object point clouds
    - epsilon: float, distance threshold for considering a point "in contact"
    - fraction_threshold: float, minimum fraction of points that must be within epsilon

    Returns:
    - list of int, timesteps where both hands are in contact
    """
    T = left_hand_sequence.shape[0]  # Number of timesteps
    contact_timesteps = []

    for t in range(T):
        # Extract point clouds at timestep t
        obj_pcd = object_sequence[t]    # Shape: (N_obj, 3)
        left_pcd = left_hand_sequence[t]  # Shape: (N_left, 3)
        right_pcd = right_hand_sequence[t] # Shape: (N_right, 3)

        # Build KDTree for the object point cloud
        tree = KDTree(obj_pcd)

        # Left hand: compute distances to nearest object point
        distances_left, _ = tree.query(left_pcd, k=1)  # k=1 for nearest neighbor
        count_left = np.sum(distances_left < epsilon)  # Number of points within epsilon
        fraction_left = count_left / left_pcd.shape[0] # Fraction of points

        # Right hand: compute distances to nearest object point
        distances_right, _ = tree.query(right_pcd, k=1)
        count_right = np.sum(distances_right < epsilon)
        fraction_right = count_right / right_pcd.shape[0]

        # Check if both hands are in contact
        if fraction_left > fraction_threshold and fraction_right > fraction_threshold:
            contact_timesteps.append(t)

    return contact_timesteps


def main():
    dev = "cuda:0"
    args = construct_args()

    with open(
        f"./data/arctic_data/data/meta/misc.json",
        "r",
    ) as f:
        misc = json.load(f)

    statcams = {}
    for sub in misc.keys():
        statcams[sub] = {
            "world2cam": torch.FloatTensor(np.array(misc[sub]["world2cam"])),
            "intris_mat": torch.FloatTensor(np.array(misc[sub]["intris_mat"])),
        }

    if args.mano_p is not None:
        mano_ps = [args.mano_p]
    else:
        mano_ps = glob(f"./data/arctic_data/data/raw_seqs/*/*.mano.npy")

    layers = construct_layers(dev)
    # object_tensor = ObjectTensors('', './arctic_data/data')
    object_tensor = ObjectTensors()
    object_tensor.to(dev)
    layers["object"] = object_tensor

    all_contact_csv = []

    pbar = tqdm(mano_ps)
    for mano_p in pbar:
        path = Path(mano_p)
        seq_name = f"{path.parent.name}/{path.name}"
        pbar.set_description("Processing %s" % mano_p)

        desc_f = mano_p.replace('data/raw_seqs', 'description').replace('.mano.npy', '/description.txt')
        try:
            task = [mano_p, dev, statcams, layers, pbar]
            out = process_seq(task, export_verts=args.export_verts, save=False)
            obj_type = path.name.split('_')[0]

            left_hand = out["world_coord"]["verts.left"]
            right_hand = out["world_coord"]["verts.right"]
            object = out["world_coord"]["verts.object"]
            contact_idxs = find_contact_timesteps(left_hand, right_hand, object)
            captions = get_captions(desc_f, obj_type, left_hand.shape[0])

            all_contact_csv.extend(
                [(seq_name, idx, cap) for idx, cap in zip(contact_idxs, captions) if cap is not None]
            )
        except Exception as e:
            logger.info(traceback.format_exc())
            logger.info(f"Failed at {mano_p}")
            continue

    df = pd.DataFrame(all_contact_csv, columns=['filename', 'timestep', 'caption'])
    df.to_csv('data/arctic_contact_all.csv', index=False)

    train_df, val_df = split_df_train_val(df, frac=0.9)
    train_df.to_csv('data/arctic_contact_train.csv', index=False)
    val_df.to_csv('data/arctic_contact_val.csv', index=False)

if __name__ == "__main__":
    main()
