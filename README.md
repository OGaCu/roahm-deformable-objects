# Hand-to-Eye Calibration (Base-to-Camera)

This repository runs a two-step pipeline to calibrate the robot base frame to the camera using an AprilTag on the end-effector: first capture poses and images while the robot moves, then compute the base-to-camera transform from those inputs.

---

## Pipeline Overview
0. **Setup:**  Close the endeffector of the arm and place the large 3D printed April tag mount on the end, make sure the PC is connected to the pandas arm and the docker container is running.
1. **Step 1:** Run `capture_poses_and_images.py` to move the robot in a figure-eight pattern and record end-effector poses and camera images at each sample.
2. **Step 2:** Run `calculate_base_to_cam.py` to load those images and poses, detect the AprilTag, and compute the mean base-to-camera transform.

Run both scripts from the **repository root**. Step 2 reads poses from `DATAPATH/poses/figure_eight_poses.npz` and images from `DATAPATH/images/` (the same `DATAPATH` used in the scripts).

---

## Step 1: Capture Poses and Images

**Script:** `capture_poses_and_images.py`

Moves the robot arm in a figure-eight pattern and captures images of the AprilTag at each sample pose.

### Command-line arguments

| Argument     | Choices      | Default  | Description                          |
|-------------|--------------|----------|--------------------------------------|
| `--camera`  | `azure`, `zed` | `azure` | Camera to use for capture.           |

### Example

```bash
# Use Azure Kinect (default)
python capture_poses_and_images.py

# Use ZED
python capture_poses_and_images.py --camera zed
```

### Outputs (saved under `DATAPATH` in the script)

- **Poses:** `poses/figure_eight_poses.npz`  
  - End-effector poses (position + quaternion) for each sample.
- **Images:** `images/image_pose_0.png`, `images/image_pose_1.png`, …  
  - One image per sample; used in step 2 for AprilTag detection.

`DATAPATH` is set inside the script (e.g. to the repo path). Ensure the `poses/` and `images/` directories exist under that path, or the script will need to create them.

---

## Step 2: Calculate Base-to-Camera Transform

**Script:** `calculate_base_to_cam.py`

Loads the captured images and poses, runs AprilTag detection (using the same camera intrinsics as in step 1), and computes the mean base-to-camera transform.

### Command-line arguments

| Argument     | Choices      | Default  | Description                                                                 |
|-------------|--------------|----------|-----------------------------------------------------------------------------|
| `--camera`  | `azure`, `zed` | `azure` | Camera used during capture; selects which intrinsics to use for detection. |

### Example

```bash
# If you used Azure in step 1 (default)
python calculate_base_to_cam.py

# If you used ZED in step 1
python calculate_base_to_cam.py --camera zed
```

Use the **same** `--camera` value as in step 1 so intrinsics match the images.

### Inputs (expected by the script)

- **Images:** `DATAPATH/images/image_pose_1.png` … `image_pose_{max_images}.png`
- **Poses:** `DATAPATH/poses/figure_eight_poses.npz` (written by step 1).

### Outputs

- **Transform:** `poses/base2cam_transform.npz`  
  - Single 4×4 SE(3) matrix: base-to-camera transform (saved under `DATAPATH` in the script).

The script also prints the mean base-to-camera matrix to the console.

---

## Dependencies

- **Python:** NumPy, OpenCV (`cv2`), SciPy (`scipy.spatial.transform`).
- **Step 1 (capture):**
  - **Robot:** `crisp_py.robot` (Robot interface).
  - **Cameras:**  
    - ZED: `pyzed.sl` (ZED SDK).  
    - Azure: `k4a` (Azure Kinect SDK).
  - **Config:** `config/control/default_cartesian_impedance.yaml` (used when crisp_py code is enabled).
- **Step 2 (calculate):**
  - **AprilTag:** `apriltag` (AprilTag detection).
  - **Intrinsics:** `azure_intrinsics` from `azure_intrinsics.py` for Azure; ZED intrinsics are defined in `apriltag_image.py`.

Install the ZED SDK / Azure Kinect SDK and the corresponding Python bindings as required for the camera you use.

---

## Summary

| Step | Script                      | Main output                          |
|------|-----------------------------|--------------------------------------|
| 1    | `capture_poses_and_images.py` | `poses/figure_eight_poses.npz`, `images/image_pose_*.png` |
| 2    | `calculate_base_to_cam.py`  | `poses/base2cam_transform.npz`       |

Use `--camera azure` or `--camera zed` consistently in both steps so intrinsics match your capture camera.
