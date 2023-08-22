"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""
import argparse
import glob
import os
import pickle as pkl

from configuration import EMDB_ROOT
from evaluation_engine import HYBRIK, EvaluationEngine
from evaluation_metrics import compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", help="Where the baseline results are stored.")
    args = parser.parse_args()

    def is_emdb1(emdb_pkl_file):
        with open(emdb_pkl_file, "rb") as f:
            data = pkl.load(f)
            return data["emdb1"]

    # Search for all the test sequences on which we evaluated the baselines in the paper.
    all_emdb_pkl_files = glob.glob(os.path.join(EMDB_ROOT, "*/*/*_data.pkl"))
    emdb1_sequence_roots = []
    for emdb_pkl_file in all_emdb_pkl_files:
        if is_emdb1(emdb_pkl_file):
            emdb1_sequence_roots.append(os.path.dirname(emdb_pkl_file))

    # Select the baselines we want to evaluate.
    baselines_to_evaluate = [HYBRIK]

    # Run the evaluation.
    evaluator_public = EvaluationEngine(compute_metrics)
    evaluator_public.run(emdb1_sequence_roots, args.result_root, baselines_to_evaluate)
