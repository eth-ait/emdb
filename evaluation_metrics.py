"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""
import cv2
import numpy as np
import torch
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.utils import local_to_global

SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])


def get_data(
    pose_gt,
    shape_gt,
    trans_gt,
    pose_hat,
    shape_hat,
    trans_hat,
    gender_gt,
    gender_hat=None,
):
    """
    Return SMPL joint positions, vertices, and global joint orientations for both ground truth and predictions.
    :param pose_gt: np array, ground-truth SMPL pose parameters including the root, shape (N, 72)
    :param shape_gt: np array, ground-truth SMPL shape parameters, shape (N, 10) or (1, 10).
    :param trans_gt: np array, ground-truth SMPL translation parameters, shape (N, 3).
    :param pose_hat: np array, predicted SMPL pose parameters including the root, shape (N, 72).
    :param shape_hat: np array, predicted SMPL shape parameters, shape (N, 10) or (1, 10).
    :param trans_hat: np array, predicted SMPL translation parameters, shape (N, 3).
    :param gender_gt: "male", "female", or "neutral"
    :param gender_hat: "male", "female", "neutral", or None (in which case it defaults ot `gender_gt`)
    :return: the predicted joints as a np array of shape (N, 24, 3)
             the ground-truth joints as a np array of shape (N, 24, 3)
             the predicted global joint orientations as a np array of shape (N, 9, 3, 3), i.e. only the major joints
             the ground-truth global joint orientations as a np array of shape (N, 24, 3, 3), i.e. all SMPL joints
             the predicted vertices as a np array of shape (N, 6890, 3)
             the ground-truth vertices as a np array of shape (N, 6890, 3)
    """
    # We use the SMPL layer and model from aitviewer for convenience.
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender_gt)

    # Create a SMPLSequence to perform the forward pass.
    smpl_seq = SMPLSequence(
        pose_gt[:, 3:],
        smpl_layer=smpl_layer,
        poses_root=pose_gt[:, :3],
        betas=shape_gt,
        trans=trans_gt,
    )

    verts_gt, jp_gt = smpl_seq.vertices, smpl_seq.joints

    # Compute the global joint orientations.
    global_oris = local_to_global(
        torch.cat([smpl_seq.poses_root, smpl_seq.poses_body], dim=-1),
        smpl_seq.skeleton[:, 0],
        output_format="rotmat",
    )

    n_frames = pose_gt.shape[0]
    glb_rot_mats_gt = global_oris.reshape((n_frames, -1, 3, 3)).detach().cpu().numpy()

    if gender_hat is None:
        gender_hat = gender_gt

    if gender_hat != gender_gt:
        smpl_layer_hat = SMPLLayer(model_type="smpl", gender=gender_hat)
    else:
        smpl_layer_hat = smpl_layer

    # Repeat the same for the predictions.
    smpl_seq_hat = SMPLSequence(
        pose_hat[:, 3:],
        smpl_layer=smpl_layer_hat,
        poses_root=pose_hat[:, :3],
        betas=shape_hat,
        trans=trans_hat,
    )
    verts_pred, jp_pred = smpl_seq_hat.vertices, smpl_seq_hat.joints
    global_oris_hat = local_to_global(
        torch.cat([smpl_seq_hat.poses_root, smpl_seq_hat.poses_body], dim=-1),
        smpl_seq_hat.skeleton[:, 0],
        output_format="rotmat",
    )

    glb_rot_mats_pred = global_oris_hat.reshape((n_frames, -1, 3, 3)).detach().cpu().numpy()
    glb_rot_mats_pred = glb_rot_mats_pred[:, SMPL_OR_JOINTS]

    return jp_pred, jp_gt, glb_rot_mats_pred, glb_rot_mats_gt, verts_pred, verts_gt


def align_by_pelvis(joints, verts=None):
    """ "Align the SMPL joints and vertices by the pelvis."""
    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices. Borrowed from
      https://github.com/aymenmir1/3dpw-eval/blob/master/evaluate.py
    :param pred_mat: np array, predicted rotation matrices, shape (N, 9, 3, 3).
    :param gt_mat: np array, ground truth rotation matrices, shape (N, 24, 3, 3).
    :return: Mean geodesic distance between input matrices.
    """
    n_frames = pred_mat.shape[0]
    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    angles_all = np.degrees(np.array(angles).reshape((n_frames, -1)))
    return np.mean(angles_all), angles_all


def compute_jitter(preds3d, gt3ds, ignored_joints_idxs=None, fps=30):
    """
    Calculate the jitter as defined in PIP paper. https://arxiv.org/pdf/2203.08528.pdf
    Code Reference: https://github.com/Xinyu-Yi/PIP/blob/main/evaluate.py
    :param preds3d: np array, ground truth joints in global space, shape (N, 24, 3).
    :param gt3ds: np array, predicted joints in global space, shape (N, 24, 3).
    :param ignored_joints_idxs: np array, SMPL joint indices to ignore, if any.
    :param fps: int, frame rate of the sequence.
    :return: mean and std. of jerk (time derivative of acceleration) of all body joints in the global space in km/s^3
    """
    if ignored_joints_idxs is None:
        ignored_joints_idxs = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    if ignored_joints_idxs is not None:
        preds3d[:, ignored_joints_idxs] = 0
        gt3ds[:, ignored_joints_idxs] = 0

    jkp = np.linalg.norm(
        (preds3d[3:] - 3 * preds3d[2:-1] + 3 * preds3d[1:-2] - preds3d[:-3]) * (fps**3),
        axis=2,
    )
    jkt = np.linalg.norm(
        (gt3ds[3:] - 3 * gt3ds[2:-1] + 3 * gt3ds[1:-2] - gt3ds[:-3]) * (fps**3),
        axis=2,
    )
    return (
        jkp.mean() / 10,
        jkp.std(axis=0).mean() / 10,
        jkt.mean() / 10,
        jkt.std(axis=0).mean() / 10,
    )


def apply_camera_transforms(joints, rotations, world2camera):
    """
    Applies camera transformations to joint locations and rotations matrices. Based on
    https://github.com/aymenmir1/3dpw-eval/blob/master/evaluate.py
    :param joints: np array, joint positions, shape (N, 24, 3).
    :param rotations: np array, joint orientations, shape (N, 24, 3, 3).
    :param world2camera: np array, the world to camera transformation, shape (N, 4, 4).
    :return: the joints after applying the camera transformation, shape (N, 24, 3)
             the orientations after applying the camera transformation, shape (N, 24, 3, 3)
    """
    joints_h = np.concatenate([joints, np.ones(joints.shape[:-1] + (1,))], axis=-1)[..., None]
    joints_c = np.matmul(world2camera[:, None], joints_h)[..., :3, 0]

    rotations_c = np.matmul(world2camera[:, None, :3, :3], rotations)

    return joints_c, rotations_c


def compute_similarity_transform(S1, S2, num_joints, verts=None):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. I.e., solves the orthogonal Procrutes problem.
    Borrowed from https://github.com/aymenmir1/3dpw-eval/blob/master/evaluate.py
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # Use only body joints for procrustes
    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    # 1. Remove mean.
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    verts_hat = None
    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {"scale": scale, "R": R, "trans": t}

    if verts_hat is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params


def compute_positional_errors(pred_joints, gt_joints, pred_verts, gt_verts, do_pelvis_alignment=True):
    """
    Computes the MPJPE and PVE errors between the predicted and ground truth joints and vertices.
    :param pred_joints: np array, predicted joints, shape (N, 24, 3).
    :param gt_joints: np array, ground truth joints, shape (N, 24, 3).
    :param pred_verts: np array, predicted vertices, shape (N, 6890, 3).
    :param gt_verts: np array, ground truth vertices, shape (N, 6890, 3).
    :param do_pelvis_alignment: bool, whether to align the predictions to the ground truth pelvis.
    :return: A dictionary with the MPJPE and MVE errors. We return the errors both with and without a PA alignment.
      Further, the dictionary contains the mean errors for this sequence, as well as the errors for each frame (_pf).
    """
    num_joints = gt_joints[0].shape[0]
    errors_jps, errors_pa_jps = [], []
    errors_verts, errors_pa_verts = [], []
    proc_rot = []

    for i, (gt3d_jps, pd3d_jps) in enumerate(zip(gt_joints, pred_joints)):
        # Get corresponding ground truth and predicted 3d joints and verts
        gt3d_jps = gt3d_jps.reshape(-1, 3)
        gt3d_verts = gt_verts[i].reshape(-1, 3)
        pd3d_verts = pred_verts[i].reshape(-1, 3)

        # Root align.
        if do_pelvis_alignment:
            gt3d_verts = align_by_pelvis(gt3d_jps, gt3d_verts)
            pd3d_verts = align_by_pelvis(pd3d_jps, pd3d_verts)
            gt3d_jps = align_by_pelvis(gt3d_jps)
            pd3d_jps = align_by_pelvis(pd3d_jps)

        # Calculate joints and verts pelvis align error
        joint_error = np.sqrt(np.sum((gt3d_jps - pd3d_jps) ** 2, axis=1))
        verts_error = np.sqrt(np.sum((gt3d_verts - pd3d_verts) ** 2, axis=1))
        errors_jps.append(np.mean(joint_error))
        errors_verts.append(np.mean(verts_error))

        # Get procrustes align error.
        pd3d_jps_sym, pd3d_verts_sym, procrustesParam = compute_similarity_transform(
            pd3d_jps, gt3d_jps, num_joints, pd3d_verts
        )
        proc_rot.append(procrustesParam["R"])

        pa_jps_error = np.sqrt(np.sum((gt3d_jps - pd3d_jps_sym) ** 2, axis=1))
        pa_verts_error = np.sqrt(np.sum((gt3d_verts - pd3d_verts_sym) ** 2, axis=1))

        errors_pa_jps.append(np.mean(pa_jps_error))
        errors_pa_verts.append(np.mean(pa_verts_error))

    result_dict = {
        "mpjpe": np.mean(errors_jps),
        "mpjpe_pa": np.mean(errors_pa_jps),
        "mve": np.mean(errors_verts),
        "mve_pa": np.mean(errors_pa_verts),
        "mat_procs": np.stack(proc_rot, 0),
        "mpjpe_pf": np.stack(errors_jps, 0),
        "mpjpe_pf_pa": np.stack(errors_pa_jps, 0),
        "mve_pf": np.stack(errors_verts, 0),
        "mve_pf_pa": np.stack(errors_pa_verts, 0),
    }

    return result_dict


def compute_metrics(
    pose_gt,
    shape_gt,
    trans_gt,
    pose_hat,
    shape_hat,
    trans_hat,
    gender_gt,
    gender_hat,
    camera_pose_gt=None,
):
    """
    Computes all the metrics we want to report.
    :param pose_gt: np array, ground-truth SMPL pose parameters including the root, shape (N, 72)
    :param shape_gt: np array, ground-truth SMPL shape parameters, shape (N, 10) or (1, 10).
    :param trans_gt: np array, ground-truth SMPL translation parameters, shape (N, 3).
    :param pose_hat: np array, predicted SMPL pose parameters including the root, shape (N, 72).
    :param shape_hat: np array, predicted SMPL shape parameters, shape (N, 10) or (1, 10).
    :param trans_hat: np array, predicted SMPL translation parameters, shape (N, 3).
    :param gender_gt: "male", "female", or "neutral"
    :param gender_hat: "male", "female", "neutral", or None (in which case it defaults ot `gender_gt`)
    :param camera_pose_gt: np array, the world to camera transformation, shape (N, 4, 4).
    :return: The function returns two dictionarys: one with the average metrics for the whole sequence and one with the
      metrics for the whole sequence put per frame.
    """
    # Get the 3D keypoints and joint angles for both ground-truth and prediction.
    pred_joints, gt_joints, pred_mats, gt_mats, pred_verts, gt_verts = get_data(
        pose_gt,
        shape_gt,
        trans_gt,
        pose_hat,
        shape_hat,
        trans_hat,
        gender_gt,
        gender_hat,
    )

    # Rotate the ground-truth joints and rotation matrices into the camera's view.
    # I.e. results of the baselines/methods should be given in camera-relative coordinates.
    if camera_pose_gt is not None:
        gt_joints, gt_mats = apply_camera_transforms(gt_joints, gt_mats, camera_pose_gt)
        gt_verts, _ = apply_camera_transforms(gt_verts, gt_mats, camera_pose_gt)

    pos_errors = compute_positional_errors(
        pred_joints * 1000.0, gt_joints * 1000.0, pred_verts * 1000.0, gt_verts * 1000.0
    )

    # Apply Procrustes rotation to the global rotation matrices.
    mats_procs_exp = np.expand_dims(pos_errors["mat_procs"], 1)
    mats_procs_exp = np.tile(mats_procs_exp, (1, len(SMPL_OR_JOINTS), 1, 1))
    mats_pred_prc = np.matmul(mats_procs_exp, pred_mats)

    # Compute differences between the predicted matrices after procrustes and GT matrices.
    mpjae_pa_final, all_angles_pa = joint_angle_error(mats_pred_prc, gt_mats)

    # Compute MPJAE without Procrustes.
    mpjae_final, all_angles = joint_angle_error(pred_mats, gt_mats)

    # Compute Jitter Error.
    jkp_mean, jkp_std, jkt_mean, jkt_std = compute_jitter(pred_joints, gt_joints)

    # These are all scalars. Choose nice names for pretty printing later.
    metrics = {
        "MPJPE [mm]": pos_errors["mpjpe"],
        "MPJPE_PA [mm]": pos_errors["mpjpe_pa"],
        "MPJAE [deg]": mpjae_final,
        "MPJAE_PA [deg]": mpjae_pa_final,
        "MVE [mm]": pos_errors["mve"],
        "MVE_PA [mm]": pos_errors["mve_pa"],
        "Jitter [km/s^3]": jkp_mean,
    }

    metrics_extra = {
        "mpjpe_all": pos_errors["mpjpe_pf"],  # (N,)
        "mpjpe_pa_all": pos_errors["mpjpe_pf_pa"],  # (N,)
        "mpjae_all": all_angles,  # (N, 9)
        "mpjae_pa_all": all_angles_pa,  # (N, 9)
        "mve_all": pos_errors["mve_pf"],  # (N,)
        "mve_pa_all": pos_errors["mve_pf_pa"],  # (N,)
        "jitter_pd": jkp_mean,  # Scalar
        "jitter_pd_std": jkp_std,  # Scalar
        "jitter_gt_mean": jkt_mean,  # Scalar
        "jitter_gt_std": jkt_std,  # Scalar
    }

    return metrics, metrics_extra
