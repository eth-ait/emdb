"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann

Script to visualize an EMDB sequence. Make sure to set the path of `EMDB_ROOT` and `SMPLX_MODELS` below.

Usage:
  python visualize.py P8 68_outdoor_handstand
"""
import argparse
import glob
import os
import pickle as pkl

import cv2
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.lines import LinesTrail
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

from configuration import (
    EMDB_ROOT,
    SMPL_SIDE_COLOR,
    SMPL_SIDE_INDEX,
    SMPL_SKELETON,
    SMPLX_MODELS,
)


def draw_kp2d(kp2d, bboxes=None):
    """Draw 2D keypoints and bounding boxes on the image with OpenCV."""

    def _draw_kp2d(img, current_frame_id):
        current_kp2d = kp2d[current_frame_id].copy()
        scale = img.shape[0] / 1000

        # Draw lines.
        for index in range(SMPL_SKELETON.shape[0]):
            i, j = SMPL_SKELETON[index]
            # color = SIDE_COLOR[max(SIDE_INDEX[i], SIDE_INDEX[j])]
            cv2.line(
                img,
                tuple(current_kp2d[i, :2].astype(np.int32)),
                tuple(current_kp2d[j, :2].astype(np.int32)),
                (0, 0, 0),
                int(scale * 3),
            )

        # Draw points.
        for jth in range(0, kp2d.shape[1]):
            color = SMPL_SIDE_COLOR[SMPL_SIDE_INDEX[jth]]
            radius = scale * 5

            out_color = (0, 0, 0)
            in_color = color

            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius * 1.4),
                out_color,
                -1,
            )
            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius),
                in_color,
                -1,
            )

        # Draw bounding box if available.
        if bboxes is not None:
            bbox = bboxes[current_frame_id]
            x_min, y_min, x_max, y_max = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        return img

    return _draw_kp2d


def draw_nothing(kp2d, bboxes=None):
    """Dummy function."""

    def _draw_nothing(img, current_frame_id):
        return img

    return _draw_nothing


def get_camera_position(Rt):
    """Get the orientation and position of the camera in world space."""
    pos = -np.transpose(Rt[:, :3, :3], axes=(0, 2, 1)) @ Rt[:, :3, 3:]
    return pos.squeeze(-1)


def get_sequence_root(args):
    """Parse the path of the sequence to be visualized."""
    sequence_id = "{:0>2d}".format(int(args.sequence))
    candidates = glob.glob(os.path.join(EMDB_ROOT, args.subject, sequence_id + "*"))
    if len(candidates) == 0:
        raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
    elif len(candidates) > 1:
        raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
    return candidates[0]


def main(args):
    # Access EMDB data.
    sequence_root = get_sequence_root(args)
    data_file = glob.glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    with open(data_file, "rb") as f:
        data = pkl.load(f)

    # Set up SMPL layer.
    gender = data["gender"]
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender)

    # Create SMPL sequence.
    smpl_seq = SMPLSequence(
        data["smpl"]["poses_body"],
        smpl_layer=smpl_layer,
        poses_root=data["smpl"]["poses_root"],
        betas=data["smpl"]["betas"].reshape((1, -1)),
        trans=data["smpl"]["trans"],
        name="EMDB Fit",
    )

    # Load 2D information.
    kp2d = data["kp2d"]
    bboxes = data["bboxes"]["bboxes"]
    drawing_function = draw_kp2d if args.draw_2d else draw_nothing

    # Load images.
    image_dir = os.path.join(sequence_root, "images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    # Load camera information.
    intrinsics = data["camera"]["intrinsics"]
    extrinsics = data["camera"]["extrinsics"]
    cols, rows = data["camera"]["width"], data["camera"]["height"]

    # Create the viewer.
    viewer_size = None
    if args.view_from_camera:
        target_height = 1080
        width = int(target_height * cols / rows)
        viewer_size = (width, target_height)

        # If we view it from the camera drawing the 3D trajectories might be disturbing, suppress it.
        args.draw_trajectories = False

    viewer = Viewer(size=viewer_size)

    # Prepare the camera.
    intrinsics = np.repeat(intrinsics[np.newaxis, :, :], len(extrinsics), axis=0)
    cameras = OpenCVCamera(intrinsics, extrinsics[:, :3], cols, rows, viewer=viewer, name="Camera")

    # Display the images on a billboard.
    raw_images_bb = Billboard.from_camera_and_distance(
        cameras,
        10.0,
        cols,
        rows,
        image_files,
        image_process_fn=drawing_function(kp2d, bboxes),
        name="Image",
    )

    # Add everything to the scene.
    viewer.scene.add(smpl_seq, cameras, raw_images_bb)

    if args.draw_trajectories:
        # Add a path trail for the SMPL root trajectory.
        smpl_path = LinesTrail(
            smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0, 0, 1, 1),
            cast_shadow=False,
            name="SMPL Trajectory",
        )

        # Add a path trail for the camera trajectory.
        # A fixed path (i.e. not a trail), could also be enabled in the GUI on the camera node by clicking "Show path".
        cam_pos = get_camera_position(extrinsics)
        camera_path = LinesTrail(
            cam_pos,
            r_base=0.003,
            color=(0.5, 0.5, 0.5, 1),
            cast_shadow=False,
            name="Camera Trajectory",
        )

        viewer.scene.add(smpl_path, camera_path)

    # Remaining viewer setup.
    if args.view_from_camera:
        # We view the scene through the camera.
        viewer.set_temp_camera(cameras)

        # Hide all the GUI controls, they can be re-enabled by pressing `ESC`.
        viewer.render_gui = False
    else:
        # We center the scene on the first frame of the SMPL sequence.
        viewer.center_view_on_node(smpl_seq)

    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    viewer.playback_fps = 30.0

    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("subject", type=str, help="The subject ID, P0 - P9.")
    parser.add_argument(
        "sequence",
        type=str,
        help="The sequence ID. This can be any unambiguous prefix of the sequence's name, i.e. for the "
        "sequence '66_outdoor_rom' it could be '66' or any longer prefix including the full name.",
    )
    parser.add_argument(
        "--view_from_camera",
        action="store_true",
        help="View it from the camera's perspective.",
    )
    parser.add_argument(
        "--draw_2d",
        action="store_true",
        help="Draw 2D keypoints and bounding boxes on the image.",
    )
    parser.add_argument(
        "--draw_trajectories",
        action="store_true",
        help="Render SMPL and camera trajectories.",
    )

    args = parser.parse_args()

    C.update_conf({"smplx_models": SMPLX_MODELS})

    main(args)
