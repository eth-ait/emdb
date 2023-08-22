"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""

import collections
import functools
import os
import pickle as pkl

import numpy as np
from tabulate import tabulate

from evaluation_loaders import load_hybrik

HYBRIK = "HybrIK"

METHOD_TO_RESULT_FOLDER = {
    HYBRIK: "hybrIK-out",
}

METHOD_TO_LOAD_FUNCTION = {
    HYBRIK: load_hybrik,
}


class EvaluationEngine(object):
    def __init__(self, metrics_compute_func, force_load=False, ignore_smpl_trans=True):
        # Function to be used to compute the metrics.
        self.compute_metrics = metrics_compute_func
        # If true, it will invalidate all caches and reload the baseline results.
        self.force_load = force_load
        # If set, the SMPL translation of the predictions will be set to 0. This only affects the jitter metric because
        # we always align either by the pelvis or via Procrustes for the other metrics.
        self.ignore_smpl_trans = ignore_smpl_trans

    def get_ids_from_sequence_root(self, sequence_root):
        res = sequence_root.split(os.path.sep)
        subject_id = res[-2]
        seq_id = res[-1]
        return subject_id, seq_id

    @functools.lru_cache()
    def _get_emdb_data(self, sequence_root):
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)
        data_file = os.path.join(sequence_root, f"{subject_id}_{seq_id}_data.pkl")
        with open(os.path.join(sequence_root, data_file), "rb") as f:
            data = pkl.load(f)
        return data

    def load_emdb_gt(self, sequence_root):
        """
        Load EMDB SMPL pose parameters.
        :param sequence_root: Where the .pkl file is stored.
        :return:
          poses_gt: a np array of shape (N, 72)
          betas_gt: a np array of shape (N, 10)
          trans_gt: a np array of shape (N, 3)
        """
        data = self._get_emdb_data(sequence_root)

        poses_body = data["smpl"]["poses_body"]
        poses_root = data["smpl"]["poses_root"]
        betas = data["smpl"]["betas"]
        trans = data["smpl"]["trans"]

        poses_gt = np.concatenate([poses_root, poses_body], axis=-1)
        betas_gt = np.repeat(betas.reshape((1, -1)), repeats=data["n_frames"], axis=0)
        trans_gt = trans

        return poses_gt, betas_gt, trans_gt

    def load_good_frames_mask(self, sequence_root):
        """Return the mask that says which frames are good and whic are not (because the human is too occluded)."""
        data = self._get_emdb_data(sequence_root)
        return data["good_frames_mask"]

    def get_gender_for_baseline(self, method):
        """Which gender to use for the baseline method."""
        if method in [HYBRIK]:
            return "neutral"
        else:
            # This will select whatever gender the ground-truth specifies.
            return None

    def compare2method(self, poses_gt, betas_gt, trans_gt, sequence_root, result_root, method):
        """Load this method's results and compute the metrics on them."""

        # Load the baseline results
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)
        method_result_dir = os.path.join(result_root, subject_id, seq_id, METHOD_TO_RESULT_FOLDER[method])
        poses_cmp, betas_cmp, trans_cmp = METHOD_TO_LOAD_FUNCTION[method](method_result_dir, self.force_load)

        if self.ignore_smpl_trans:
            trans_cmp = np.zeros_like(trans_cmp)

        # Load camera parameters.
        data = self._get_emdb_data(sequence_root)
        world2cam = data["camera"]["extrinsics"]

        gender_gt = data["gender"]
        gender_hat = self.get_gender_for_baseline(method)

        # For some frames there is too much occlusion, we ignore these.
        good_frames_mask = self.load_good_frames_mask(sequence_root)

        metrics, metrics_extra = self.compute_metrics(
            poses_gt[good_frames_mask],
            betas_gt[good_frames_mask],
            trans_gt[good_frames_mask],
            poses_cmp[good_frames_mask],
            betas_cmp[good_frames_mask],
            trans_cmp[good_frames_mask],
            gender_gt,
            gender_hat,
            world2cam[good_frames_mask],
        )

        return metrics, metrics_extra, method

    def evaluate_single_sequence(self, sequence_root, result_root, methods):
        """Evaluate a single sequence for all methods."""
        ms, ms_extra, ms_names = [], [], []

        poses_gt, betas_gt, trans_gt = self.load_emdb_gt(sequence_root)

        for method in methods:
            m, m_extra, ms_name = self.compare2method(poses_gt, betas_gt, trans_gt, sequence_root, result_root, method)

            ms.append(m)
            ms_extra.append(m_extra)
            ms_names.append(ms_name)

        return ms, ms_extra, ms_names

    def to_pretty_string(self, metrics, model_names):
        """Print the metrics onto the console, but pretty."""
        if not isinstance(metrics, list):
            metrics = [metrics]
            model_names = [model_names]
        assert len(metrics) == len(model_names)
        headers, rows = [], []
        for i in range(len(metrics)):
            values = []
            for k in metrics[i]:
                if i == 0:
                    headers.append(k)
                values.append(metrics[i][k])
            rows.append([model_names[i]] + values)
        return tabulate(rows, headers=["Model"] + headers)

    def run(self, sequence_roots, result_root, methods):
        """Run the evaluation on all sequences and all methods."""
        if not isinstance(sequence_roots, list):
            sequence_roots = [sequence_roots]

        # For every baseline, accumulate the metrics of all frames so that we can later compute statistics on them.
        ms_all = None
        ms_names = None
        n_frames = 0
        for sequence_root in sequence_roots:
            ms, ms_extra, ms_names = self.evaluate_single_sequence(sequence_root, result_root, methods)

            print("Metrics for sequence {}".format(sequence_root))
            print(self.to_pretty_string(ms, ms_names))

            n_frames += ms_extra[0]["mpjpe_all"].shape[0]

            if ms_all is None:
                ms_all = [collections.defaultdict(list) for _ in ms]

            for i in range(len(ms)):
                ms_all[i]["mpjpe_all"].append(ms_extra[i]["mpjpe_all"])
                ms_all[i]["mpjpe_pa_all"].append(ms_extra[i]["mpjpe_pa_all"])
                ms_all[i]["mpjae_all"].append(np.mean(ms_extra[i]["mpjae_all"], axis=-1))  # Mean over joints.
                ms_all[i]["mpjae_pa_all"].append(np.mean(ms_extra[i]["mpjae_pa_all"], axis=-1))  # Mean over joints.
                ms_all[i]["jitter_pd"].append(ms_extra[i]["jitter_pd"])
                if "mve_all" in ms_extra[i]:
                    ms_all[i]["mve_all"].append(ms_extra[i]["mve_all"])
                if "mve_pa_all" in ms_extra[i]:
                    ms_all[i]["mve_pa_all"].append(ms_extra[i]["mve_pa_all"])

        # Compute the mean and std over all sequences.
        ms_all_agg = []
        for i in range(len(ms_all)):
            mpjpe_all = np.concatenate(ms_all[i]["mpjpe_all"], axis=0)
            mpjpe_pa_all = np.concatenate(ms_all[i]["mpjpe_pa_all"], axis=0)
            mpjae_all = np.concatenate(ms_all[i]["mpjae_all"], axis=0)
            mpjae_pa_all = np.concatenate(ms_all[i]["mpjae_pa_all"], axis=0)
            jitter_all = np.array(ms_all[i]["jitter_pd"])
            metrics = {
                "MPJPE [mm]": np.mean(mpjpe_all),
                "MPJPE std": np.std(mpjpe_all),
                "MPJPE_PA [mm]": np.mean(mpjpe_pa_all),
                "MPJPE_PA std": np.std(mpjpe_pa_all),
                "MPJAE [deg]": np.mean(mpjae_all),
                "MPJAE std": np.std(mpjae_all),
                "MPJAE_PA [deg]": np.mean(mpjae_pa_all),
                "MPJAE_PA std": np.std(mpjae_pa_all),
                "Jitter [10m/s^3]": np.mean(jitter_all),
                "Jitter std": np.std(jitter_all),
            }
            if "mve_all" in ms_all[i]:
                mve_all = np.concatenate(ms_all[i]["mve_all"], axis=0)
                metrics["MVE [mm]"] = np.mean(mve_all)
                metrics["MVE std"] = np.std(mve_all)
            if "mve_pa_all" in ms_all[i]:
                mve_pa_all = np.concatenate(ms_all[i]["mve_pa_all"], axis=0)
                metrics["MVE_PA [mm]"] = np.mean(mve_pa_all)
                metrics["MVE_PA std"] = np.std(mve_pa_all)
            ms_all_agg.append(metrics)

        print("Metrics for all sequences")
        print(self.to_pretty_string(ms_all_agg, ms_names))
        print(" ")

        print("Total Number of Frames:", n_frames)
