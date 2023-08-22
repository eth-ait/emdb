# EMDB Dataset Structure

After unzipping the downloaded files, please arrange their contents as follows:

```python
EMDB_ROOT
├── P0
├── P1
├── ...
├── P9
```

Each participant's folder looks like this:

```
EMDB_ROOT
├── PX
    ├── sequence1
        ├── images
        ├── {PX}_{sequence1}_data.pkl
        ├── {PX}_{sequence1}_video.mp4
    ├── sequence2
    ├── ...
    ├── sequenceN
```

The `images` subfolder contains the raw images as single files in named in the format `{%5d}.jpg` from 0. The video file shows a side-by-side view of the original RGB images and the EMDB reference overlaid on top. The contents of the pickle file are described in the following.


## `*_data.pkl`

The pickle file contains a dictionary with the following keys:

| Key | Type | Description |
| --- | --- | --- |
| `gender` | `string` | Either `"female"`, `"male"`, or `"neutral"`. |
| `name` | `string` | The name of this sequence in the format `{subject_id}_{sequence_id}`. |
| `emdb1` | `bool` | Whether this sequence is part of EMDB 1 as mentioned in the paper. We evaluated state-of-the-art RGB-based baselines on EMDB 1. |
| `emdb2` | `bool` | Whether this sequence is part of EMDB 2 as mentioned in the paper. We evaluated GLAMR's global trajectories on EMDB 2. |
| `n_frames` | `int` | Length of this sequence. |
| `good_frames_mask` | `np.ndarray`| Of shape `(n_frames, )` indicating which frames are considered valid (`True`) and which aren't. The invalid frames are hand-selected instances where the person is out-of-view or occluded entirely. |
| `camera` | `dict` | Camera information (see below). |
| `smpl` | `dict` | SMPL parameters (see below). |
| `kp2d` | `np.ndarray` | Of shape `(n_frames, 24, 2)` containing the SMPL joints projected into the camera. |
| `bboxes` | `dict` | Bounding box data (see below). |

### `camera`

| Key | Type | Description |
| --- | --- | --- |
| `intrinsics` | `np.ndarray` | Of shape `(3, 3)` containing the camera intrinsics. |
| `extrinsics` | `np.ndarray` | Of shape `(n_frames, 4, 4)` containing the camera extrinsics. |
| `width` | `int` | The width of the image. |
| `height` | `int` | The height of the image. |

### `smpl`

| Key | Type | Description |
| --- | --- | --- |
| `poses_root` | `np.ndarray` | Of shape `(n_frames, 3)` containing the SMPL root orientation. |
| `poses_body` | `np.ndarray` | Of shape `(n_frames, 69)` containing the SMPL pose parameters. |
| `trans` | `np.ndarray` | Of shape `(n_frames, 3)` containing the SMPL root translation. |
| `betas` | `np.ndarray` | Of shape `(10, )` containing the SMPL shape parameters. |


### `bboxes`

| Key | Type | Description |
| --- | --- | --- |
| `bboxes` | `np.ndarray` | Of shape `(n_frames, 4)` containing the 2D bounding boxes in the format `(x_min, y_min, x_max, y_max)`. |
| `invalid_idxs` | `np.ndarray` | Indexes of invalid bounding boxes. A bounding box at frame `i` is invalid if `good_frames_mask[i] == False` or if `x_max - x_min <= 0 or y_max - y_min <= 0`. |