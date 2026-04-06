
# Single Arm Capture

`single_arm_capture.py` captures Azure Kinect frames and end-effector poses while the single arm runs a joint-trajectory from a `.task` file. Need to run the arm via docker first with ```cd move_some_robots/crisp/crisp_controllers_demos/
ROBOT_IP=192.168.2.2 docker compose up launch_franka```.

## Purpose

Use this script to:
- connect to the arm
- run a joint-trajectory using a `.task` file
- stream Azure color/depth frames and EE poses
- save frames, poses, and videos for calibration/analysis

## Usage

```bash
python single_arm_capture.py --seq-name <sequence_name>
```

Options:
- `--seq-name` (required): Name of the capture sequence. Output goes to `DATAPATH/captured_data_single_arm/<seq-name>`.
- `--task` (optional): Path to a `.task` file with joint waypoints. Default: `DATAPATH/right_traj.task`.
- `--stream-hz` (optional): Target streaming rate (Hz) for image+pose capture. Default: `30.0`.
- `--waypoints-per-chunk` (optional): Save data every N waypoints and continue. Output: `chunk_0/`, `chunk_1/`, ... with `rgbd.npz`, poses, and videos per chunk.
- `--max-waypoints` (optional): Upper limit on number of waypoints to execute.
- `--start-index` (optional): 0-based index of the first waypoint to use.
- `--end-index` (optional): 0-based one-past-last waypoint index to use (default: full length).

Notes:
- `DATAPATH` is hardcoded in the script: `/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects`.
- The script uses Azure Kinect only (no Zed option).
- Joint trajectory only (no cartesian mode).
- The script aligns the image frames with the pose frames with the images currently having a 7 frame delay

## Output

Outputs are saved under:
`/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/captured_data_single_arm/<seq-name>`

Files and folders:
- `rgbd.npz`: Stacked arrays with keys `color` and optionally `depth`.
- `rgbd.npz` `color`: `(N, H, W, 3)` uint8 color frames.
- `rgbd.npz` `depth`: `(N, H, W)` uint16 depth (only if depth is available).
- `single_arm_poses.npz`: EE pose vectors `[x, y, z, qx, qy, qz, qw]` at capture time.
- `video.mp4`: Color video at 30 fps.
- `depth.mp4`: Colorized depth video at 30 fps (only if depth is available).

Chunked output (when using `--waypoints-per-chunk`):
- `chunk_0/`, `chunk_1/`, ... each containing the same file set listed above.

## Notes

- Make sure the Azure Kinect is connected and powered before starting capture.
- The camera config (resolution, depth mode, FPS) is set in `single_arm_capture.py`.

# Double Arm Capture

`double_arm_capture.py` captures frames and end-effector poses while both arms run a trajectory.

## Purpose

Use this script to:
- connect to both arms
- run a joint or cartesian trajectory
- stream camera frames and left/right EE poses
- save frames, poses, and videos for calibration/analysis

## Usage

```bash
python double_arm_capture.py --seq-name <sequence_name>
```

Options:
- `--seq-name` (required): Name of the capture sequence. Output goes to `OUTPUT_ROOT/captured_data_double_arm/<seq-name>`.
- `--camera` (optional): Camera to use: `azure` or `zed`. Default: `azure`.
- `--trajectory` (optional): `joint` or `cartesian`. Default: `joint`.
- `--task-left` (optional): Path to left arm `.task`. Default: `DATAPATH/left_traj.task`.
- `--task-right` (optional): Path to right arm `.task`. Default: `DATAPATH/right_traj.task`.
- `--stream-hz` (optional): Target streaming rate (Hz). Default: `30.0`.
- `--waypoints-per-chunk` (optional): Joint mode only. Save data every N waypoints and continue. Output: `chunk_0/`, `chunk_1/`, ... with `rgbd.npz`, poses, and videos per chunk.
- `--max-waypoints` (optional): Joint mode only. Upper limit on number of waypoints to execute.
- `--start-index` (optional): Joint mode only. 0-based index of the first waypoint to use.
- `--end-index` (optional): Joint mode only. 0-based one-past-last waypoint index to use (default: full length).

Notes:
- `DATAPATH` is hardcoded in the script: `/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects`.
- `OUTPUT_ROOT` is hardcoded in the script: `/mnt/mydisk`.
- `azure` captures RGB-D. `zed` captures RGB only.
- `cartesian` uses pose waypoints from the `.task` files. Chunking options apply to joint mode only.
- The script aligns the image frames with the pose frames with the images currently having a 7 frame delay.

## Output

Outputs are saved under:
`/mnt/mydisk/captured_data_double_arm/<seq-name>`

Files and folders:
- `rgbd.npz`: Stacked arrays with keys `color` and optionally `depth`.
- `rgbd.npz` `color`: `(N, H, W, 3)` uint8 color frames.
- `rgbd.npz` `depth`: `(N, H, W)` uint16 depth (only if depth is available).
- `left_arm_poses.npz`: Left EE pose vectors `[x, y, z, qx, qy, qz, qw]`.
- `right_arm_poses.npz`: Right EE pose vectors `[x, y, z, qx, qy, qz, qw]`.
- `video.mp4`: Color video at 30 fps.
- `depth.mp4`: Colorized depth video at 30 fps (only if depth is available).

Chunked output (when using `--waypoints-per-chunk`):
- `chunk_0/`, `chunk_1/`, ... each containing the same file set listed above.

## Notes

- Make sure the camera is connected and both arms are ready before starting capture.
- The camera config (resolution, depth mode, FPS) is set in `double_arm_capture.py`.
