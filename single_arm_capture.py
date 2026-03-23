"""Moves the robot arm along a trajectory and streams (camera image, EE pose) pairs.

=== High-level flow ===
1. Trajectory is predefined: waypoints from a .task file (joint angles) or cartesian poses.
   - Joint mode: waypoints have no inherent "fps"; the robot interpolates over time_to_goal per segment.
   - Cartesian mode: the main loop sends set_target at ~50 Hz; the robot follows continuously.
2. A separate streaming thread runs at --stream-hz (default 10): each tick it captures one image and
   reads the current EE pose, then saves the pair. So you get (image, pose) pairs at the stream rate.
3. Trajectory "fps" and stream rate are independent: the trajectory is defined by waypoints and
   controller timing; the stream rate is how often we sample (image, pose). Frames and poses are
   buffered in RAM during the run; all saves to disk happen after motion and streaming finish, so
   actual capture rate can reach camera FPS (e.g. 30 fps) without disk I/O as a bottleneck.

=== Order of pose vs camera capture (per stream tick) ===
- Azure: pose_before → get_capture() [image acquired] → pose_after → we store average(pose_before, pose_after)
  and the image. So the stored pose approximates the EE pose at capture time.
- Zed: grab() [image acquired] → then read EE pose. So the stored pose is the pose *after* the frame.
"""

import argparse
import json
import threading
from pathlib import Path
import time

import numpy as np
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import pyzed.sl as sl
import cv2
import k4a
from helper import densify_waypoints, move_to_nonblocking, weighted_average_transforms, PoseSE3



def parse_task_poses(task_path: str | Path, save_npz: str | Path | None = None) -> list[np.ndarray]:
    """Parse a .task file and extract poses as a list of 4x4 transformation matrices.

    The .task file is JSON from a Franka/teaching workflow; poses are stored under
    parameter.parameterized_poses[].pose_with_joint_angles.pose (16 floats, row-major 4x4).

    Args:
        task_path: Path to the .task file.
        save_npz: If set, save the list of transforms to this .npz file (arr_0, arr_1, ...).

    Returns:
        List of 4x4 numpy arrays (SE3 transformation matrices).
    """
    task_path = Path(task_path)
    with open(task_path, "r") as f:
        data = json.load(f)
    parameterized = data.get("parameter", {}).get("parameterized_poses", [])
    transforms = []
    for item in parameterized:
        pwja = item.get("pose_with_joint_angles") or item
        pose = pwja.get("pose")
        if pose is None or len(pose) != 16:
            continue
        T = np.array(pose, dtype=np.float64).reshape(4, 4).T
        transforms.append(T)
    if save_npz is not None:
        np.savez(save_npz, *transforms)
    return transforms


def parse_task_joints(task_path: str | Path) -> list[np.ndarray]:
    """Parse a .task file and extract joint angles for each waypoint.

    The .task file is JSON from a Franka/teaching workflow; joint angles are under
    parameter.parameterized_poses[].pose_with_joint_angles.joint_angles (7 floats for Franka).

    Args:
        task_path: Path to the .task file.

    Returns:
        List of 1D numpy arrays of joint positions (radians), one per waypoint.
    """
    task_path = Path(task_path)
    with open(task_path, "r") as f:
        data = json.load(f)
    parameterized = data.get("parameter", {}).get("parameterized_poses", [])
    joint_configs = []
    for item in parameterized:
        pwja = item.get("pose_with_joint_angles") or item
        q = pwja.get("joint_angles")
        if q is None:
            continue
        joint_configs.append(np.array(q, dtype=np.float64))
    return joint_configs


def _streaming_capture_loop(
    capture_active: threading.Event,
    left_arm,
    camera: str,
    zed_cam,
    zed_image,
    zed_runtime_params,
    azure_device,
    azure_transformation,
    pose_list: list,
    frame_list: list,
    lock: threading.Lock,
    rate_hz: float,
    pose_list_second: list = None,
) -> None:
    """
    Run at target rate_hz: each tick = one (image, EE pose) pair appended to lists. No disk I/O
    during the loop so rate is limited only by camera FPS and get_capture(); saves happen after the run.
    Order per tick:
      - Zed:   grab image → retrieve → read EE pose → append (frame, pose).
      - Azure: read EE pose (before) → get_capture() → read EE pose (after) → average pose →
               depth_image_to_color_camera() → append (color_bgr, depth_aligned, pose).
    """
    period = 1.0 / rate_hz
    while capture_active.is_set():
        t0 = time.time()
        try:
            if camera == "zed":
                if zed_cam.grab(zed_runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(period)
                    continue
                zed_cam.retrieve_image(zed_image, sl.VIEW.LEFT)
                frame = zed_image.get_data().copy()  # copy so next grab doesn't overwrite
                p = left_arm.end_effector_pose
                pose_vec = np.array([
                    p.position[0], p.position[1], p.position[2],
                    p.orientation.as_quat()[0], p.orientation.as_quat()[1],
                    p.orientation.as_quat()[2], p.orientation.as_quat()[3],
                ])
                with lock:
                    frame_list.append({"color": frame, "depth": None})
                    pose_list.append(pose_vec)
            else:
                p_first = left_arm.end_effector_pose.copy()
                capture = azure_device.get_capture(-1)
                p_second = left_arm.end_effector_pose.copy()
                p_avg = weighted_average_transforms(p_first, p_second, 0.5, 0.5)
                ## For testing purposes we are captrunig both poses and returning them
                pose_vec_first= np.array([
                    p_first.position[0], p_first.position[1], p_first.position[2],
                    p_first.orientation.as_quat()[0], p_first.orientation.as_quat()[1],
                    p_first.orientation.as_quat()[2], p_first.orientation.as_quat()[3],
                ])
                pose_vec_second = np.array([
                    p_second.position[0], p_second.position[1], p_second.position[2],
                    p_second.orientation.as_quat()[0], p_second.orientation.as_quat()[1],
                    p_second.orientation.as_quat()[2], p_second.orientation.as_quat()[3],
                ])
                color_bgr = cv2.cvtColor(capture.color.data, cv2.COLOR_BGRA2BGR).copy()
                # Depth transformed to color camera (same resolution as color, pixel-aligned)
                if capture.depth is not None and azure_transformation is not None:
                    depth_color_img = azure_transformation.depth_image_to_color_camera(capture.depth)
                    depth_data = depth_color_img.data.copy()  # (H, W) uint16, same H,W as color
                else:
                    depth_data = None
                with lock:
                    frame_list.append({"color": color_bgr, "depth": depth_data})
                    pose_list.append(pose_vec_first)
                    if pose_list_second is not None:
                        pose_list_second.append(pose_vec_second)
        except Exception as e:
            print(f"Streaming capture error: {e}")
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


parser = argparse.ArgumentParser(description="Capture poses and images with specified camera.")
parser.add_argument(
    "--camera",
    type=str,
    choices=["azure", "zed"],
    default="azure",
    help="Camera to use: 'azure' or 'zed' (default: zed)",
)
parser.add_argument(
    "--trajectory",
    type=str,
    choices=["cartesian", "joint"],
    default="cartesian",
    help="Use cartesian waypoints (set_target) or joint waypoints from .task (joint_trajectory_controller)",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Path to .task file for waypoints (default: DATAPATH/right_traj.task for joint, right_traj_2.task for cartesian)",
)
parser.add_argument(
    "--seq-name",
    type=str,
    required=True,
    help="Name of this capture sequence (used to create DATAPATH/captured_data_single_arm/{seq_name}).",
)
parser.add_argument(
    "--stream-hz",
    type=float,
    default=30.0,
    help="Target streaming rate in Hz (image + EE pose per tick). For 30 fps use 30 and ensure Azure camera_fps=FPS_30; actual rate limited by camera, get_capture, and disk I/O.",
)
args = parser.parse_args()
camera = args.camera
trajectory_mode = args.trajectory
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"
TASK_PATH = args.task or (f"{DATAPATH}/right_traj.task" if trajectory_mode == "joint" else f"{DATAPATH}/right_traj_2.task")
seq_name = args.seq_name
base_dir = Path(DATAPATH) / "captured_data_single_arm" / seq_name
frames_dir = base_dir / "frames"


# initialize robot
left_arm = Robot(namespace="")
left_arm.wait_until_ready()

print("Going to home position...")
left_arm.home()

# Setup Params and waypoints
if trajectory_mode == "joint":
    joint_waypoints = parse_task_joints(TASK_PATH)
    if not joint_waypoints:
        raise RuntimeError(f"No joint waypoints found in {TASK_PATH}")
    print(f"Loaded {len(joint_waypoints)} joint waypoints from {TASK_PATH}")
    left_arm.controller_switcher_client.switch_controller("joint_trajectory_controller")
else:
    # move_to() and set_target(pose) need cartesian_impedance_controller
    left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    left_arm.cartesian_controller_parameters_client.load_param_config(
        file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/clipped_cartesian_impedance.yaml"
    )
    left_arm.cartesian_controller_parameters_client.set_parameters([
        ("task.error_clip.x", 0.005), ("task.error_clip.y", 0.005), ("task.error_clip.z", 0.005),
        ("task.error_clip.rx", 0.008), ("task.error_clip.ry", 0.008), ("task.error_clip.rz", 0.008),
    ])
    raw_waypoints = parse_task_poses(TASK_PATH)
    MAX_WAYPOINT_STEP_M = 0.15
    waypoints = densify_waypoints(raw_waypoints, max_translation_step=MAX_WAYPOINT_STEP_M)
    print("len(waypoints): ", len(waypoints))
    if not waypoints:
        raise RuntimeError(f"No valid waypoints found in {TASK_PATH}")
    target_pose = left_arm.end_effector_pose.copy()
    target_pose.position = waypoints[0][0:3, 3]
    target_pose.orientation = Rotation.from_matrix(waypoints[0][0:3, 0:3])
    waypoint_index = 0

print("Starting capture...")



# Setup camera
if camera == "zed":
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30
    zed.open(init_params)
    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    azure_transformation = None
else:
    # -------------------------------------------------------------------------
    # Azure Kinect: configure here. Actual capture rate is capped by camera_fps.
    # See: https://github.com/etiennedub/pyk4a#device-configuration
    # -------------------------------------------------------------------------
    print("Using Azure camera")
    device = k4a.Device.open()
    if device is None:
        exit(-1)
    device_config = k4a.DeviceConfiguration(
        # Color: format and resolution (e.g. RES_720P, RES_1080P, RES_1440P, RES_1536P, RES_2160P)
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_720P,
        # Depth: e.g. OFF, NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
        # FPS: 5, 15, or 30. This caps max frames you can get; stream_hz should be <= this.
        camera_fps=k4a.EFramesPerSecond.FPS_30,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False,
    )
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        exit(-1)
    # Transformation for depth_image_to_color_camera (depth aligned to color resolution)
    cal = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    azure_transformation = k4a.Transformation.create(cal)

# --- Streaming capture (buffer in RAM, save to disk after motion is done for true 30 fps) ---
pose_list = []
pose_list_second = []
frame_list = []  # each item: {"color": bgr_array, "depth": array or None}
pose_list_lock = threading.Lock()
capture_active = threading.Event()
stream_hz = args.stream_hz
TIME_TO_GOAL = 3.0  # seconds per joint waypoint when using joint trajectory

if trajectory_mode == "joint":
    capture_active.set()
    stream_thread = threading.Thread(
        target=_streaming_capture_loop,
        kwargs=dict(
            capture_active=capture_active,
            left_arm=left_arm,
            camera=camera,
            zed_cam=zed if camera == "zed" else None,
            zed_image=image if camera == "zed" else None,
            zed_runtime_params=runtime_params if camera == "zed" else None,
            azure_device=device if camera == "azure" else None,
            azure_transformation=azure_transformation if camera == "azure" else None,
            pose_list=pose_list,
            pose_list_second=pose_list_second,
            frame_list=frame_list,
            lock=pose_list_lock,
            rate_hz=stream_hz,
        ),
        daemon=True,
    )
    stream_thread.start()
    joint_names = left_arm.config.joint_names
    for i, q in enumerate(joint_waypoints[:5]):
        left_arm.joint_trajectory_controller_client.send_joint_config(
            joint_names, q.tolist(), time_to_goal=TIME_TO_GOAL, blocking=False
        )
        time.sleep(TIME_TO_GOAL)
        print(f"Joint waypoint {i + 1}/{len(joint_waypoints)} sent")
    time.sleep(0.5)  # allow last motion to settle
    capture_active.clear()
    stream_thread.join(timeout=2.0)
    print(f"Streaming capture finished: {len(pose_list)} camera–EE pairs (buffered in RAM)")
else:
    target_pose.position = waypoints[0][0:3, 3]
    target_pose.orientation = Rotation.from_matrix(waypoints[0][0:3, 0:3])
    waypoint_index = 0
    print("target_pose_rotation:", target_peose)
    left_arm.move_to(pose=target_pose, speed=0.15)

    capture_active.set()
    stream_thread = threading.Thread(
        target=_streaming_capture_loop,
        kwargs=dict(
            capture_active=capture_active,
            left_arm=left_arm,
            camera=camera,
            zed_cam=zed if camera == "zed" else None,
            zed_image=image if camera == "zed" else None,
            zed_runtime_params=runtime_params if camera == "zed" else None,
            azure_device=device if camera == "azure" else None,
            azure_transformation=azure_transformation if camera == "azure" else None,
            pose_list=pose_list,
            pose_list_second=pose_list_second,
            frame_list=frame_list,
            lock=pose_list_lock,
            rate_hz=stream_hz,
        ),
        daemon=True,
    )
    stream_thread.start()

    start_time = 0.0
    threshold = 0.1
    prev_pose = left_arm.end_effector_pose.copy()
    while waypoint_index < len(waypoints):
        end_time = time.time()
        print(f"Loop ~{1/(end_time - start_time):.0f} Hz | position error: {sum(left_arm.end_effector_pose.position - target_pose.position):.4f}")
        start_time = time.time()

        if abs(sum(left_arm.end_effector_pose.position - target_pose.position)) < threshold:
            waypoint_index += 1
            if waypoint_index < len(waypoints):
                target_pose.position = waypoints[waypoint_index][0:3, 3]
                target_pose.orientation = Rotation.from_matrix(waypoints[waypoint_index][0:3, 0:3])
                print(f"Waypoint {waypoint_index} reached")
            threshold = 0.1
        elif abs(sum(left_arm.end_effector_pose.position - prev_pose.position)) < 0.001:
            threshold += 0.001
        prev_pose = left_arm.end_effector_pose.copy()

        left_arm.set_target(pose=target_pose)
        time.sleep(0.02)  # ~50 Hz control loop

    capture_active.clear()
    stream_thread.join(timeout=2.0)
    print(f"Streaming capture finished: {len(pose_list)} camera–EE pairs (buffered in RAM)")

# Save all buffered frames and poses to disk (no disk I/O during capture → true 30 fps)
n_saved = len(pose_list)
if n_saved != len(frame_list):
    print(f"Warning: pose_list length ({n_saved}) != frame_list length ({len(frame_list)}); saving min.")
    n_saved = min(n_saved, len(frame_list))

base_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving {n_saved} frames and poses to {base_dir}...")

# 1) Save all PNG frames under .../captured_data_single_arm/{seq_name}/frames
for i in range(n_saved):
    cv2.imwrite(str(frames_dir / f"single_arm_image_{i}.png"), frame_list[i]["color"])

# 2) Stack RGB-D into a single rgbd.npz: color (N,H,W,3), depth (N,H,W) if available
colors = np.stack([frame_list[i]["color"] for i in range(n_saved)], axis=0)  # (N,H,W,3)
if camera == "azure":
    depth_list = [frame_list[i].get("depth") for i in range(n_saved)]
    any_depth = any(d is not None for d in depth_list)
    if any_depth:
        first_valid = next((d for d in depth_list if d is not None), None)
        if first_valid is not None:
            H_d, W_d = first_valid.shape
            depth_stack = np.zeros((n_saved, H_d, W_d), dtype=first_valid.dtype)
            for i, d in enumerate(depth_list):
                if d is not None:
                    depth_stack[i] = d
        else:
            depth_stack = None
    else:
        depth_stack = None
else:
    depth_stack = None

rgbd_path = base_dir / "rgbd.npz"
if depth_stack is not None:
    np.savez(rgbd_path, color=colors, depth=depth_stack)
else:
    np.savez(rgbd_path, color=colors)

# 3) Save single-arm poses under .../captured_data_single_arm/{seq_name}
# np.savez(base_dir / "single_arm_poses.npz", *pose_list[:n_saved])
np.savez(base_dir / "single_arm_poses.npz", *pose_list[:n_saved])
np.savez(base_dir / "single_arm_poses_second.npz", *pose_list_second[:n_saved])

# 4) Save color video.mp4 and depth.mp4 in .../captured_data_single_arm/{seq_name}
H, W, _ = frame_list[0]["color"].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = base_dir / "video.mp4"
video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (W, H))
for i in range(n_saved):
    video_writer.write(frame_list[i]["color"])
video_writer.release()

if depth_stack is not None:
    depth_video_path = base_dir / "depth.mp4"
    depth_writer = cv2.VideoWriter(str(depth_video_path), fourcc, 30.0, (W, H))
    max_depth_mm = 1500.0
    for i in range(n_saved):
        d = depth_stack[i].astype(np.float32)
        d_norm = np.clip(d / max_depth_mm, 0.0, 1.0)
        d_img = (d_norm * 255.0).astype(np.uint8)
        d_color = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
        depth_writer.write(d_color)
    depth_writer.release()

print("Done saving to disk.")

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done drawing a circle!")


print("return to home and shutdown")
left_arm.home()
left_arm.shutdown()
