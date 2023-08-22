"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""
import glob
import os
import pickle as pkl

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_hybrik(result_root, force_load):
    """Load HybrIK results."""
    hybrik_dir = result_root
    hybrik_cache_dir = os.path.join(hybrik_dir, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "romp-out.npz")

    if not os.path.exists(hybrik_cache_file) or force_load:
        hybrik_betas, hybrik_poses_rot, hybrik_trans = [], [], []
        for pkl_file in sorted(glob.glob(os.path.join(hybrik_dir, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                hybrik_data = pkl.load(f)

            hybrik_poses_rot.append(hybrik_data["pred_theta_mats"].reshape((1, -1, 3, 3)))
            hybrik_betas.append(hybrik_data["pred_shape"])

            # NOTE This is not the SMPL translation, it's a translation added to the outputs of SMPL
            #  but this does not matter because we align to the root, except for the jitter metric.
            hybrik_trans.append(hybrik_data["transl"])

        hybrik_poses_rot = np.concatenate(hybrik_poses_rot, axis=0)
        hybrik_poses = R.as_rotvec(R.from_matrix(hybrik_poses_rot.reshape((-1, 3, 3)))).reshape(
            hybrik_poses_rot.shape[0], -1
        )
        hybrik_betas = np.concatenate(hybrik_betas, axis=0)
        hybrik_trans = np.concatenate(hybrik_trans, axis=0)

        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            hybrik_poses=hybrik_poses,
            hybrik_betas=hybrik_betas,
            hybrik_trans=hybrik_trans,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        hybrik_poses = hybrik_results["hybrik_poses"]
        hybrik_betas = hybrik_results["hybrik_betas"]
        hybrik_trans = hybrik_results["hybrik_trans"]

    return hybrik_poses, hybrik_betas, hybrik_trans
