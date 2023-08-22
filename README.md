# EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild

_Official Repository for the ICCV 2023 paper [EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild]()_. 

_Authors: Manuel Kaufmann, Jie Song, Chen Guo, Kaiyue Shen, Tianjian Jiang, Chengcheng Tang, Juan Zarate, Otmar Hilliges_

## [Project Page](https://ait.ethz.ch/emdb) | [Paper]() | [Supplementary]() | [Video]() | [Data](https://emdb.ait.ethz.ch)

<img src="https://files.ait.ethz.ch/projects/emdb/assets/teaser.jpg"/> 

## Dataset
To receive access to the data, please fill out the [application form](https://emdb.ait.ethz.ch). You will receive an e-mail with more information after your application has been approved.

For an overview of how EMDB is structured, please refer to the [dataset overview](dataset.md).

## Visualization
We use [aitviewer](https://github.com/eth-ait/aitviewer) to visualize the data.

### Installation
```bash
pip install aitviewer
```
This does not automatically install a GPU-version of PyTorch. If your environment already contains it, you should be good to go, otherwise you may wish to install it manually.

Please also download the SMPL model by following the instructions on the [SMPL-X GitHub page](https://github.com/vchoutas/smplx#downloading-the-model).

### Setup
1. Change the `SMPLX_MODELS` variable in `configuration.py` to where you downloaded the SMPL model. This folder should contain a subfolder `smpl` with the respective model files.
2. Change the `EMDB_ROOT` variable in `configuration.py` to where you extracted the EMDB dataset. This folder should containt subfolders `P0`, `P1`, etc. 

### Visualize EMDB Data
Run the following command to visualize a sequence. `SUBJECT_ID` refers to the ID Of the participant, i.e. `P0-P9` and `SEQUENCE_ID` is the 2-digit identifier that is prepended to each sequence's name:

```python
python visualize.py --subject {SUBJECT_ID} --sequence {SEQUENCE_ID}
```

By default, this opens the viewer in the 3D view. You can choose to show the reprojected poses instead by specifying `--view_from_camera`. If you specify `--draw_2d` the 2D keypoints and bounding boxes will be drawn on top of the image. If you pass `--draw_trajectories` the SMPL root and camera trajectories will be drawn in addition.

### Visualize GLAMR
We provide a script to visualize [GLAMR](https://github.com/NVlabs/GLAMR) results. An example result that reproduces Figure 9 of the main paper is provided in `assets/GLAMR`. To visualize it, run the following command:

```python
python visualize_GLAMR.py
```

## Evaluation

### Example using HybrIK
We provide code to load and evaluate HybrIK results on the EMDB test set, which in the paper is referred to as `EMDB 1`. Based on this evaluation code, it should be straight-forward to extend the evaluation to other methods (see below).

To run the evaluation, use the following command:
```python
python evaluate.py {RESULT_ROOT}
```

The `RESULT_ROOT` is a folder that is expected to have the same general folder structure as EMDB, i.e.:
```
RESULT_ROOT
├── PX
    ├── sequence1
        ├── hybrIK-out
            ├── 000000.pkl
            ├── 000001.pkl
            ├── ...
    ├── sequence2
    ├── ...
    ├── sequenceN
```

I.e., `EMDB_ROOT` can function as a `RESULT_ROOT`, if the corresponding results are stored in a subfolder `hybrIK-out` for each sequence. The evaluation code computes the MPJPE, MPJAE, MVE, and jitter metrics as reported in the paper. It reports both the pelvis-aligned and Procrustes-aligned versions (*-PA), as well as standard deviations. Further, it prints the metrics for each sequence individually, as well as the average over all sequences.

### How to evaluate your own method
In order to run the evaluations with your own results, follow these steps:
1. In [`evaluation_loaders.py`](evaluation_loaders.py) define a function to load your result. Follow the signature and return values of the existing `load_hybrik` function.
2. In [`evaluation_engine.py`](evaluation_engine.py) register your method by giving it a name in a global variable, e.g. `MYMETHOD = 'My Method'`. Then using that name as a key, extend the following two `dict`s (follow the existing example with HybrIK for reference):
    - `METHOD_TO_RESULT_FOLDER`: This maps to the subfolder in `{RESULT_ROOT}/{SUBJECT_ID}/{SEQUENCE_ID}` where your methods result will be stored.
    - `METHOD_TO_LOAD_FUNCTION`: This maps to the loading function you defined in step 1.
3. In the function [`EvaluationEngine.get_gender_for_baseline`](evaluation_engine.py) select the appropriate SMPL gender for your method.
4. Finally, in [`evaluate.py`](evaluate.py) import `MYMETHOD` and add it to the list of methods that the evaluation engine should evaluate.

## Citation
If you use this code or data, please cite the following paper:
```bibtex
@inproceedings{kaufmann2023emdb,
  author = {Kaufmann, Manuel and Song, Jie and Guo, Chen and Shen, Kaiyue and Jiang, Tianjian and Tang, Chengcheng and Z{\'a}rate, Juan Jos{\'e} and Hilliges, Otmar},
  title = {{EMDB}: The {E}lectromagnetic {D}atabase of {G}lobal 3{D} {H}uman {P}ose and {S}hape in the {W}ild},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```

## Contact
For any questions or problems, please open an issue or contact [Manuel Kaufmann](mailto:manuel.kaufmann@inf.ethz.ch).
