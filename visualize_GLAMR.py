"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann

Script to visualize an example GLAMR result. This reproduces Fig. 9 of the paper.

Usage:
  python visualize_GLAMR.py
"""
import glob
import os
import pickle

import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

from configuration import EMDB_ROOT


def cam2world(Rt):
    new_Rt = np.eye(4)[np.newaxis].repeat(repeats=Rt.shape[0], axis=0)
    pos = -np.transpose(Rt[:, :3, :3], axes=(0, 2, 1)) @ Rt[:, :3, 3:]
    rot = np.copy(np.transpose(Rt[:, :3, :3], axes=(0, 2, 1)))
    rot[:, :3, 1:3] *= -1.0
    new_Rt[:, :3, :3] = rot
    new_Rt[:, :3, 3:] = pos
    return new_Rt


def world2cam(Rt):
    new_Rt = np.eye(4)[np.newaxis].repeat(repeats=Rt.shape[0], axis=0)
    new_Rt[:, :3, :3] = Rt[:, :3, :3]
    new_Rt[:, :3, 1:3] *= -1.0
    new_Rt[:, :3, :3] = np.transpose(new_Rt[:, :3, :3], axes=(0, 2, 1))
    new_Rt[:, :3, 3:] = -new_Rt[:, :3, :3] @ Rt[:, :3, 3:]
    return new_Rt


if __name__ == "__main__":
    # Load GLAMR output.
    glamr_file = "assets/GLAMR/P9_80_seed1.pkl"
    with open(glamr_file, "rb") as f:
        glamr_data = pickle.load(f)

    person = glamr_data["person_data"][0]
    poses_glamr = person["smpl_pose"]
    betas_glamr = person["smpl_beta"]
    trans_glamr = person["root_trans_world"]
    ori_glamr = person["smpl_orient_world"]
    Rt_pd = glamr_data["cam_pose"]
    K_pd = person["cam_K"]

    # Load EMDB ground-truth data.
    sequence_root = os.path.join(os.path.join(EMDB_ROOT, "P9", "80_outdoor_walk_big_circle"))
    gt_data_file = glob.glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    with open(gt_data_file, "rb") as f:
        gt_data = pickle.load(f)

    poses_gt = gt_data["smpl"]["poses_body"]
    ori_gt = gt_data["smpl"]["poses_root"]
    betas_gt = gt_data["smpl"]["betas"]
    betas_gt = np.repeat(betas_gt.reshape((1, -1)), repeats=gt_data["n_frames"], axis=0)
    trans_gt = gt_data["smpl"]["trans"]
    cols, rows = gt_data["camera"]["width"], gt_data["camera"]["height"]
    Rt_gt = gt_data["camera"]["extrinsics"]
    K_gt = gt_data["camera"]["intrinsics"][np.newaxis].repeat(Rt_gt.shape[0], axis=0)

    # Align the two camera trajectories at the first frame.
    Rt_pd_w = cam2world(Rt_pd)
    Rt_gt_w = cam2world(Rt_gt)
    assert Rt_pd_w.shape == Rt_gt_w.shape

    # Compute the relative transformation between the two cameras.
    R_pd_f0 = Rt_pd_w[0, :3, :3]
    T_pd_f0 = Rt_pd_w[0, :3, 3:]
    R_gt_f0 = Rt_gt_w[0, :3, :3]
    T_gt_f0 = Rt_gt_w[0, :3, 3:]

    R_rel = R_gt_f0 @ R_pd_f0.T
    T_rel = T_gt_f0 - R_gt_f0 @ R_pd_f0.T @ T_pd_f0
    Rt_rel = np.eye(4)
    Rt_rel[:3, :3] = R_rel
    Rt_rel[:3, 3:] = T_rel
    Rt_rel = Rt_rel[np.newaxis].repeat(repeats=Rt_pd_w.shape[0], axis=0)

    # Apply the relative transformation to the predicted camera trajectory.
    Rt_pd_aligned_w = Rt_rel @ Rt_pd_w
    Rt_pd_aligned = world2cam(Rt_pd_aligned_w)

    # Define a helper function so that we can treat the root joint position the same way GLAMR does it and apply the
    # alignment that we found above.
    def post_fk_glamr(vertices, joints, align=False):
        # Subtract the position of the root joint from all vertices and joint positions and add the root translation.
        t = trans_glamr[:]
        cur_root_trans = joints[:, [0], :]
        vertices = vertices - cur_root_trans + t[:, None, :]
        joints = joints - cur_root_trans + t[:, None, :]

        def _to_h(x):
            return np.concatenate([x, np.ones(shape=(x.shape[:-1]) + (1,))], axis=-1)

        def _apply_transform(x, R):
            x_h = _to_h(x)
            return np.matmul(R[:, None], x_h[..., None]).squeeze(-1)[..., :3]

        if align:
            vertices_r = _apply_transform(vertices, Rt_rel)
            joints_r = _apply_transform(joints, Rt_rel)
            return vertices_r, joints_r

        return vertices, joints

    # Instantiate an SMPL sequence for the GLAMR data so that we can perform a forward pass through the SMPL model.
    # We set z_up=True because GLAMR data is using z_up coordinates.
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    smpl_sequence_glamr = SMPLSequence(
        poses_body=poses_glamr,
        poses_root=ori_glamr,
        betas=betas_glamr,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=(149 / 255, 149 / 255, 149 / 255, 0.8),
        z_up=True,
        name="GLAMR",
    )

    # Align the GLAMR result with the transformation that we found above.
    vs, js = smpl_sequence_glamr.vertices, smpl_sequence_glamr.joints
    vs_ori, js_ori = post_fk_glamr(vs, js)

    glamr_color = (68 / 255, 115 / 255, 23 / 255, 1.0)
    vs_aligned, js_aligned = post_fk_glamr(vs, js, align=True)

    # Prepare renderables to be displayed in the viewer.
    glamr_meshes_aligned = Meshes(
        vs_aligned,
        smpl_sequence_glamr.faces,
        name="GLAMR SMPL Prediction",
        color=glamr_color,
    )

    # We also render the trajectories explicitly. We use thick lines because the trajectories are long and they
    # wouldn't otherwise be visible very well.
    # If you want the trajectories to be built up progressively, use a `LinesTrail` instead.
    glamr_root_aligned = Lines(
        js_aligned[:, 0],
        r_base=0.06,
        mode="line_strip",
        color=(1, 0, 0, 1),
        cast_shadow=False,
        name="GLAMR SMPL Root Trajectory",
    )
    glamr_root_aligned.n_frames = js_aligned[:, 0].shape[0]

    glamr_cam_aligned = Lines(
        Rt_pd_aligned_w[:, :3, 3],
        r_base=0.04,
        mode="line_strip",
        color=(180 / 255, 180 / 255, 180 / 255, 1),
        cast_shadow=False,
        name="GLAMR Camera Trajectory",
    )
    glamr_cam_aligned.n_frames = Rt_pd_aligned_w[:, :3, 3].shape[0]

    # Instantiate a SMPL sequence for the EMDB ground-truth.
    smpl_sequence_gt = SMPLSequence(
        poses_body=poses_gt,
        poses_root=ori_gt,
        betas=betas_gt,
        trans=trans_gt,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=(160 / 255, 160 / 255, 160 / 255, 1.0),
        z_up=False,
        name="EMDB SMPL Ground Truth",
    )

    gt_root = Lines(
        smpl_sequence_gt.joints[:, 0],
        r_base=0.06,
        mode="line_strip",
        color=(0, 0, 1, 1),
        cast_shadow=False,
        name="EMDB SMPL Root Trajectory",
    )
    gt_root.n_frames = smpl_sequence_gt.joints[:, 0].shape[0]

    gt_cam = Lines(
        Rt_gt_w[:, :3, 3],
        r_base=0.04,
        mode="line_strip",
        color=(0, 0, 0, 1),
        cast_shadow=False,
        name="EMDB Camera Trajectory",
    )
    gt_cam.n_frames = Rt_gt_w[:, :3, 3].shape[0]

    # Create the viewer.
    viewer = Viewer()

    # We instantiate the GLAMR and EMDB cameras as actual OpenCVCameras so that the scene can be viewed from the
    # camera's perspective if desired. The thick trajectory lines obstruct the view from the camera, but they
    # can be disabled in the GUI.
    gt_opencv_cam = OpenCVCamera(K_gt, Rt_gt[:, :3, :], cols, rows, viewer=viewer, name="EMDB Camera")
    glamr_opencv_cam = OpenCVCamera(K_pd, Rt_pd_aligned[:, :3, :], cols, rows, viewer=viewer, name="GLAMR Camera")

    # Add everything to the scene
    viewer.scene.add(glamr_meshes_aligned, smpl_sequence_gt)
    viewer.scene.add(glamr_root_aligned, gt_root)
    viewer.scene.add(glamr_cam_aligned, gt_cam)
    viewer.scene.add(glamr_opencv_cam, gt_opencv_cam)

    # Set initial viewer camera.
    viewer.center_view_on_node(glamr_meshes_aligned)

    # Other viewer settings
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.playback_fps = 30.0
    viewer.shadows_enabled = False
    viewer.auto_set_camera_target = False

    viewer.run()
    viewer.close()
