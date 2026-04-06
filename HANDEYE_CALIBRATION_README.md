# Camera Calibration

## Overview

capture_poses_and_images_for_calibration moves a robot arm in a figure-eight trajectory while capturing end-effector poses and camera images for calibration purposes (specifically hand-to-eye calibration using an AprilTag).

calculate_base_to_cam then computes the transformation from robot base frame to camera frame using: the captured robot poses from the figure-eight trajectory and the detected AprilTag poses from images.

This performs a hand–eye calibration  computation and outputs a single averaged transformation matrix.

It supports two camera types:

* **Azure Kinect** (RGB + color-aligned depth)
* **ZED camera** (RGB only in this script)

The collected data is saved in a structured format for downstream calibration pipelines.

---

## Usage

```bash
python capture_poses_and_images_for_calibration_right.py --seq-name <sequence_name> [--camera azure|zed]
```

``` bash
python calculate_base_to_cam.py \
    --calib-seq-name <sequence_name> \
    [--camera azure|zed] \
    [--side left|right]
```

### Arguments

`capture_poses_and_images_for_calibration_{side}.py`

* `--seq-name` (required):
  Name of the calibration sequence. Data will be saved under:
  `captured_calibration_data/<seq-name>/`
* `--camera` (optional, default=`azure`):
  Camera type to use



`calculate_base_to_cam.py`

* `--calib-seq-name` (required):
Name of the dataset created by the capture script
(under `captured_calibration_data/<seq-name>/`)
 * `--camera` (default=azure):
Must match the camera used during data capture
* `--side` (default=right):
Which arm dataset to use

---

## Data Storage

All data is saved under:

```
captured_calibration_data/<seq-name>/
├── frames/
│   ├── calibration_right_image_0.png
│   ├── calibration_right_image_1.png
│   └── ...
├── right_calibration_poses.npz
└── right_calibration_rgbd.npz

captured_calibration_data/<seq-name>/base2cam_transform_<side>.npz
```

#### Pose File

`right_calibration_poses.npz`

Array of poses:
* shape: (N, 7)
* format: [x, y, z, qx, qy, qz, qw]


#### RGB-D File

`right_calibration_rgbd.npz`

Contains:

* color: (N, H, W, 3)
* depth: (N, H, W) (Azure only, optional)

---

## Notes

* **Pose/Image Alignment**:

  * Each saved pose corresponds to a captured frame
  * Data is appended in the same order

* **Depth Data (Azure only)**:

  * Depth is aligned to the color camera frame
  * Missing depth frames are handled gracefully


---