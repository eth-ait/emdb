"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann

Define a few constants that we need throughout.
"""
import numpy as np

# TODO update paths to match your setup
EMDB_ROOT = "V:/emdb"
SMPLX_MODELS = "D:/data/smplx_models"

SMPL_SKELETON = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [2, 5],
        [5, 8],
        [8, 11],
        [1, 4],
        [4, 7],
        [7, 10],
        [3, 6],
        [6, 9],
        [9, 14],
        [9, 13],
        [9, 12],
        [12, 15],
        [14, 17],
        [17, 19],
        [19, 21],
        [21, 23],
        [13, 16],
        [16, 18],
        [18, 20],
        [20, 22],
    ]
)

# fmt: off
# 0 center, 1 right, 2 left
SMPL_SIDE_INDEX = [0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2, 1, 2, 1, 2, 1]
# fmt: on

SMPL_SIDE_COLOR = [
    (255, 0, 255),
    (0, 0, 255),
    (255, 0, 0),
]
