"""Dual-arm trajectory capture: one streaming thread (camera + left/right EE poses), main thread runs motion.

Same pattern as single_arm_capture: buffer (image, left_pose, right_pose) in RAM at --stream-hz;
save all to disk after motion and streaming finish. Supports cartesian waypoints (pose) or joint
waypoints (--trajectory joint) from .task files.
"""

import argparse
import json
import threading
from pathlib import Path

import numpy as np
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import pyzed.sl as sl
import cv2
import time
import k4a
from helper import densify_waypoints, weighted_average_transforms


def parse_task_poses(task_path: str | Path, save_npz: str | Path | None = None) -> list[np.ndarray]:
    """Parse a .task file and extract poses as a list of 4x4 transformation matrices."""
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
    """Parse a .task file and extract joint angles per waypoint (7 floats for Franka)."""
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


def _apply_frame_pose_delay(
    frame_list: list,
    left_pose_list: list,
    right_pose_list: list,
    delay: int,
) -> tuple[list, list, list]:
    if delay <= 0:
        return frame_list, left_pose_list, right_pose_list

    if len(frame_list) <= delay:
        frame_list = []
    else:
        frame_list = frame_list[delay:]

    if len(left_pose_list) <= delay:
        left_pose_list = []
    else:
        left_pose_list = left_pose_list[:-delay]

    if len(right_pose_list) <= delay:
        right_pose_list = []
    else:
        right_pose_list = right_pose_list[:-delay]

    return frame_list, left_pose_list, right_pose_list


def _streaming_capture_loop_double(
    capture_active: threading.Event,
    left_arm,
    right_arm,
    camera: str,
    zed_cam,
    zed_image,
    zed_runtime_params,
    azure_device,
    azure_transformation,
    frame_list: list,
    left_pose_list: list,
    right_pose_list: list,
    lock: threading.Lock,
    rate_hz: float,
) -> None:
    """
    Run at rate_hz: each tick = one image + left EE pose + right EE pose (buffered, no disk I/O).
    Azure: pose before (both arms) → get_capture() → pose after (both) → average poses, color-aligned depth.
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
                frame = zed_image.get_data().copy()
                p_left = left_arm.end_effector_pose
                p_right = right_arm.end_effector_pose
                left_vec = np.array([
                    p_left.position[0], p_left.position[1], p_left.position[2],
                    p_left.orientation.as_quat()[0], p_left.orientation.as_quat()[1],
                    p_left.orientation.as_quat()[2], p_left.orientation.as_quat()[3],
                ])
                right_vec = np.array([
                    p_right.position[0], p_right.position[1], p_right.position[2],
                    p_right.orientation.as_quat()[0], p_right.orientation.as_quat()[1],
                    p_right.orientation.as_quat()[2], p_right.orientation.as_quat()[3],
                ])
                with lock:
                    frame_list.append({"color": frame, "depth": None})
                    left_pose_list.append(left_vec)
                    right_pose_list.append(right_vec)
            else:
                p_left_first = left_arm.end_effector_pose.copy()
                p_right_first = right_arm.end_effector_pose.copy()
                capture = azure_device.get_capture(-1)
                p_left_second = left_arm.end_effector_pose.copy()
                p_right_second = right_arm.end_effector_pose.copy()
                p_left_avg = weighted_average_transforms(p_left_first, p_left_second, 0.5, 0.5)
                p_right_avg = weighted_average_transforms(p_right_first, p_right_second, 0.5, 0.5)
                left_vec = np.array([
                    p_left_avg.position[0], p_left_avg.position[1], p_left_avg.position[2],
                    p_left_avg.as_quat()[0], p_left_avg.as_quat()[1],
                    p_left_avg.as_quat()[2], p_left_avg.as_quat()[3],
                ])
                right_vec = np.array([
                    p_right_avg.position[0], p_right_avg.position[1], p_right_avg.position[2],
                    p_right_avg.as_quat()[0], p_right_avg.as_quat()[1],
                    p_right_avg.as_quat()[2], p_right_avg.as_quat()[3],
                ])
                color_bgr = cv2.cvtColor(capture.color.data, cv2.COLOR_BGRA2BGR).copy()
                if capture.depth is not None and azure_transformation is not None:
                    depth_color_img = azure_transformation.depth_image_to_color_camera(capture.depth)
                    depth_data = depth_color_img.data.copy()
                else:
                    depth_data = None
                with lock:
                    frame_list.append({"color": color_bgr, "depth": depth_data})
                    left_pose_list.append(left_vec)
                    right_pose_list.append(right_vec)
        except Exception as e:
            print(f"Streaming capture error: {e}")
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


def _save_chunk(
    frame_list: list,
    left_pose_list: list,
    right_pose_list: list,
    chunk_dir: Path,
    camera: str,
) -> int:
    """Save one chunk to chunk_dir: rgbd.npz, left/right_arm_poses.npz, video.mp4, depth.mp4 (no PNG frames).

    Within each chunk, all indexing is 0 to n_saved-1: rgbd color[i], depth[i], left_arm_poses arr_i, right_arm_poses arr_i.
    Returns n_saved.
    """
    n_saved = min(len(frame_list), len(left_pose_list), len(right_pose_list))
    if n_saved == 0:
        return 0
    chunk_dir.mkdir(parents=True, exist_ok=True)
    colors = np.stack([frame_list[i]["color"] for i in range(n_saved)], axis=0)
    depth_stack = None
    if camera == "azure":
        depth_list = [frame_list[i].get("depth") for i in range(n_saved)]
        if any(d is not None for d in depth_list):
            first_valid = next((d for d in depth_list if d is not None), None)
            if first_valid is not None:
                H_d, W_d = first_valid.shape
                depth_stack = np.zeros((n_saved, H_d, W_d), dtype=first_valid.dtype)
                for i, d in enumerate(depth_list):
                    if d is not None:
                        depth_stack[i] = d
    if depth_stack is not None:
        np.savez(chunk_dir / "rgbd.npz", color=colors, depth=depth_stack)
    else:
        np.savez(chunk_dir / "rgbd.npz", color=colors)
    np.savez(chunk_dir / "left_arm_poses.npz", *left_pose_list[:n_saved])
    np.savez(chunk_dir / "right_arm_poses.npz", *right_pose_list[:n_saved])
    H, W = frame_list[0]["color"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(chunk_dir / "video.mp4"), fourcc, 30.0, (W, H))
    for i in range(n_saved):
        video_writer.write(frame_list[i]["color"])
    video_writer.release()
    if depth_stack is not None:
        depth_writer = cv2.VideoWriter(str(chunk_dir / "depth.mp4"), fourcc, 30.0, (W, H))
        max_depth_mm = 1500.0
        for i in range(n_saved):
            d = depth_stack[i].astype(np.float32)
            d_norm = np.clip(d / max_depth_mm, 0.0, 1.0)
            d_img = (d_norm * 255.0).astype(np.uint8)
            depth_writer.write(cv2.applyColorMap(d_img, cv2.COLORMAP_JET))
        depth_writer.release()

    print(
        f"Chunk saved to {chunk_dir}: "
        f"color shape={colors.shape}, "
        f"depth shape={None if depth_stack is None else depth_stack.shape}, "
        f"left_poses={len(left_pose_list[:n_saved])}, right_poses={len(right_pose_list[:n_saved])}"
    )
    return n_saved


parser = argparse.ArgumentParser(description="Dual-arm capture: stream camera + left/right EE poses, save after motion.")
parser.add_argument("--camera", type=str, choices=["azure", "zed"], default="azure", help="Camera: 'azure' or 'zed'")
parser.add_argument(
    "--seq-name",
    type=str,
    required=True,
    help="Name of this capture sequence (used to create DATAPATH/captured_data_double_arm/{seq_name}).",
)
parser.add_argument(
    "--trajectory",
    type=str,
    choices=["cartesian", "joint"],
    default="joint",
    help="cartesian: pose waypoints (set_target). joint: joint waypoints from .task (joint_trajectory_controller).",
)
parser.add_argument(
    "--task-left",
    type=str,
    default=None,
    help="Path to .task for left arm (default: DATAPATH/left_traj.task)",
)
parser.add_argument(
    "--task-right",
    type=str,
    default=None,
    help="Path to .task for right arm (default: DATAPATH/right_traj.task)",
)
parser.add_argument(
    "--stream-hz",
    type=float,
    default=30.0,
    help="Target streaming rate in Hz (buffer in RAM; save after run).",
)
parser.add_argument(
    "--waypoints-per-chunk",
    type=int,
    default=None,
    help="For joint mode: save data every N waypoints, clear RAM, then continue. Enables long runs (e.g. 340 waypoints). "
    "Output: chunk_0/, chunk_1/, ... (rgbd.npz, poses, video.mp4, depth.mp4 per chunk; no PNG frames, no merge). None = single run (default).",
)
parser.add_argument(
    "--max-waypoints",
    type=int,
    default=None,
    help="For joint mode: optional upper limit on number of waypoints to execute (e.g. stop after 200 even if task has more).",
)
parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help="For joint mode: 0-based index of first waypoint in the .task to use for capture.",
)
parser.add_argument(
    "--end-index",
    type=int,
    default=None,
    help="For joint mode: 0-based one-past-last waypoint index to use from the .task (default: full length).",
)
args = parser.parse_args()
camera = args.camera
trajectory_mode = args.trajectory
stream_hz = args.stream_hz
waypoints_per_chunk = args.waypoints_per_chunk
max_waypoints = args.max_waypoints
start_index = args.start_index
end_index = args.end_index
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"
OUTPUT_ROOT = Path("/mnt/mydisk")
seq_name = args.seq_name
base_dir = OUTPUT_ROOT / "captured_data_double_arm" / seq_name
TASK_LEFT = args.task_left or f"{DATAPATH}/left_traj.task"
TASK_RIGHT = args.task_right or f"{DATAPATH}/right_traj.task"

# Robots
left_arm = Robot(namespace="left")
right_arm = Robot(namespace="right")
left_arm.wait_until_ready()
right_arm.wait_until_ready()

print("Going to home position...")
# left_arm.home()
# right_arm.home()

TIME_TO_GOAL = 1.5  # seconds per joint waypoint when using joint trajectory
# Delay between captured robot poses and captured images (camera capture is non-blocking).
IMAGE_POSE_FRAME_DELAY = 7

if trajectory_mode == "joint":
    left_joint_waypoints = parse_task_joints(TASK_LEFT)
    right_joint_waypoints = parse_task_joints(TASK_RIGHT)
    if not left_joint_waypoints:
        raise RuntimeError(f"No joint waypoints in {TASK_LEFT}")
    if not right_joint_waypoints:
        raise RuntimeError(f"No joint waypoints in {TASK_RIGHT}")
    total_joint_waypoints = min(len(left_joint_waypoints), len(right_joint_waypoints))
    if len(left_joint_waypoints) != len(right_joint_waypoints):
        print(
            f"Note: left has {len(left_joint_waypoints)} waypoints, right has {len(right_joint_waypoints)}; "
            f"using first {total_joint_waypoints} for both."
        )
    # Clamp requested segment [start_index, end_index) to available range
    if start_index < 0:
        start_index = 0
    if end_index is None or end_index > total_joint_waypoints:
        end_index = total_joint_waypoints
    if start_index >= end_index:
        raise ValueError(
            f"Invalid waypoint segment: start-index={start_index}, end-index={end_index}, "
            f"available range is [0, {total_joint_waypoints})."
        )
    left_joint_waypoints = left_joint_waypoints[start_index:end_index]
    right_joint_waypoints = right_joint_waypoints[start_index:end_index]
    n_joint_waypoints = len(left_joint_waypoints)
    # Optional additional cap on the selected segment
    if max_waypoints is not None and max_waypoints > 0 and max_waypoints < n_joint_waypoints:
        print(f"Limiting joint waypoints in segment from {n_joint_waypoints} to max-waypoints={max_waypoints}.")
        n_joint_waypoints = max_waypoints
        left_joint_waypoints = left_joint_waypoints[:n_joint_waypoints]
        right_joint_waypoints = right_joint_waypoints[:n_joint_waypoints]
    print(
        f"Joint mode: using waypoint segment [{start_index}, {end_index}) from .task "
        f"({n_joint_waypoints} waypoints per arm after max-waypoints)."
    )
    left_arm.controller_switcher_client.switch_controller("joint_trajectory_controller")
    right_arm.controller_switcher_client.switch_controller("joint_trajectory_controller")
    left_waypoints = None
    right_waypoints = None
else:
    left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    left_arm.cartesian_controller_parameters_client.load_param_config(
        file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/clipped_cartesian_impedance.yaml"
    )
    left_arm.cartesian_controller_parameters_client.set_parameters([
        ("task.error_clip.x", 0.003), ("task.error_clip.y", 0.003), ("task.error_clip.z", 0.003),
        ("task.error_clip.rx", 0.008), ("task.error_clip.ry", 0.008), ("task.error_clip.rz", 0.008),
    ])
    right_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    right_arm.cartesian_controller_parameters_client.load_param_config(
        file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/clipped_cartesian_impedance.yaml"
    )
    right_arm.cartesian_controller_parameters_client.set_parameters([
        ("task.error_clip.x", 0.005), ("task.error_clip.y", 0.005), ("task.error_clip.z", 0.005),
        ("task.error_clip.rx", 0.008), ("task.error_clip.ry", 0.008), ("task.error_clip.rz", 0.008),
    ])
    left_raw_waypoints = parse_task_poses(TASK_LEFT)
    right_raw_waypoints = parse_task_poses(TASK_RIGHT)
    MAX_WAYPOINT_STEP_M = 0.03
    left_waypoints = densify_waypoints(left_raw_waypoints, max_translation_step=MAX_WAYPOINT_STEP_M)
    right_waypoints = densify_waypoints(right_raw_waypoints, max_translation_step=MAX_WAYPOINT_STEP_M)
    if not left_waypoints:
        raise RuntimeError(f"No valid waypoints in {TASK_LEFT}")
    if not right_waypoints:
        raise RuntimeError(f"No valid waypoints in {TASK_RIGHT}")

# Camera
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
    print("Using Azure camera")
    device = k4a.Device.open()
    if device is None:
        exit(-1)
    device_config = k4a.DeviceConfiguration(
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_720P,
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
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
    cal = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    azure_transformation = k4a.Transformation.create(cal)

# Buffers and capture thread
frame_list = []
left_pose_list = []
right_pose_list = []
capture_lock = threading.Lock()
capture_active = threading.Event()

print("Starting capture...")

use_chunks = trajectory_mode == "joint" and waypoints_per_chunk is not None and waypoints_per_chunk > 0
if use_chunks:
    chunk_size = waypoints_per_chunk
    num_chunks = (n_joint_waypoints + chunk_size - 1) // chunk_size
    print(f"Chunked joint mode: {n_joint_waypoints} waypoints in {num_chunks} chunks of up to {chunk_size}.")

# for chunk_idx in range(num_chunks if use_chunks else 1):
for chunk_idx in range(num_chunks if use_chunks else 1):
    cancel_chunk_requested = threading.Event()  # used in joint chunked mode to cancel mid-chunk
    if use_chunks:
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_joint_waypoints)
        frame_list.clear()
        left_pose_list.clear()
        right_pose_list.clear()
        print(f"Chunk {chunk_idx + 1}/{num_chunks}: waypoints {chunk_start}..{chunk_end - 1}")
    else:
        chunk_start = 0
        chunk_end = n_joint_waypoints if trajectory_mode == "joint" else 0

    capture_active.set()
    stream_thread = threading.Thread(
        target=_streaming_capture_loop_double,
        kwargs=dict(
            capture_active=capture_active,
            left_arm=left_arm,
            right_arm=right_arm,
            camera=camera,
            zed_cam=zed if camera == "zed" else None,
            zed_image=image if camera == "zed" else None,
            zed_runtime_params=runtime_params if camera == "zed" else None,
            azure_device=device if camera == "azure" else None,
            azure_transformation=azure_transformation,
            frame_list=frame_list,
            left_pose_list=left_pose_list,
            right_pose_list=right_pose_list,
            lock=capture_lock,
            rate_hz=stream_hz,
        ),
        daemon=True,
    )
    stream_thread.start()

    if trajectory_mode == "joint":
        left_joint_names = left_arm.config.joint_names
        right_joint_names = right_arm.config.joint_names
        # Optional: allow cancelling this chunk midway (chunked mode only)
        # if use_chunks:
        #     def _wait_for_cancel():
        #         try:
        #             input()  # blocks until Enter; then set event
        #         except (EOFError, KeyboardInterrupt):
        #             pass
        #         cancel_chunk_requested.set()

        #     print(f"Chunk {chunk_idx}: Press Enter at any time to cancel this chunk early.")
        #     cancel_thread = threading.Thread(target=_wait_for_cancel, daemon=True)
        #     cancel_thread.start()

        for i in range(chunk_start, chunk_end):
            # if use_chunks and cancel_chunk_requested.is_set():
            #     print(f"Chunk {chunk_idx} cancelled by user after waypoint {i}.")
            #     break
            left_arm.joint_trajectory_controller_client.send_joint_config(
                left_joint_names, left_joint_waypoints[i].tolist(), time_to_goal=TIME_TO_GOAL, blocking=False
            )
            right_arm.joint_trajectory_controller_client.send_joint_config(
                right_joint_names, right_joint_waypoints[i].tolist(), time_to_goal=TIME_TO_GOAL, blocking=False
            )
            time.sleep(TIME_TO_GOAL)
            print(f"Joint waypoint {i + 1}/{n_joint_waypoints} sent")
        time.sleep(0.5)  # allow last motion to settle
    else:
        # Cartesian mode (runs once, not chunked)
        left_target_pose = left_arm.end_effector_pose.copy()
        right_target_pose = right_arm.end_effector_pose.copy()
        left_target_pose.position = left_waypoints[0][0:3, 3]
        left_target_pose.orientation = Rotation.from_matrix(left_waypoints[0][0:3, 0:3])
        right_target_pose.position = right_waypoints[0][0:3, 3]
        right_target_pose.orientation = Rotation.from_matrix(right_waypoints[0][0:3, 0:3])
        left_waypoint_index = 0
        right_waypoint_index = 0
        threshold = 0.05
        left_arm.move_to(pose=left_target_pose, speed=0.15)
        right_arm.move_to(pose=right_target_pose, speed=0.15)

        while left_waypoint_index < len(left_waypoints) and right_waypoint_index < len(right_waypoints):
            if abs(sum(left_arm.end_effector_pose.position - left_target_pose.position)) < threshold:
                left_waypoint_index += 1
                if left_waypoint_index < len(left_waypoints):
                    left_target_pose.position = left_waypoints[left_waypoint_index][0:3, 3]
                    left_target_pose.orientation = Rotation.from_matrix(left_waypoints[left_waypoint_index][0:3, 0:3])
                    print(f"Left waypoint {left_waypoint_index} reached")
            if abs(sum(right_arm.end_effector_pose.position - right_target_pose.position)) < threshold:
                right_waypoint_index += 1
                if right_waypoint_index < len(right_waypoints):
                    right_target_pose.position = right_waypoints[right_waypoint_index][0:3, 3]
                    right_target_pose.orientation = Rotation.from_matrix(right_waypoints[right_waypoint_index][0:3, 0:3])
                    print(f"Right waypoint {right_waypoint_index} reached")

            left_arm.set_target(pose=left_target_pose)
            right_arm.set_target(pose=right_target_pose)
            time.sleep(0.02)  # ~50 Hz control loop

    capture_active.clear()
    stream_thread.join(timeout=2.0)
    frame_list, left_pose_list, right_pose_list = _apply_frame_pose_delay(
        frame_list, left_pose_list, right_pose_list, IMAGE_POSE_FRAME_DELAY
    )
    n_saved = min(len(frame_list), len(left_pose_list), len(right_pose_list))
    print(f"Chunk/run finished: {n_saved} frames, {len(left_pose_list)} left poses, {len(right_pose_list)} right poses")

    if use_chunks:
        # Ask interactively whether to save this chunk or discard it (partial if cancelled early)
        cancelled_early = use_chunks and cancel_chunk_requested.is_set()
        if n_saved > 0:
            prompt = (
                f"Save chunk {chunk_idx} (waypoints {chunk_start}..{chunk_end - 1}, {n_saved} frames"
                + ("; partial - chunk was cancelled" if cancelled_early else "")
                + ")? [y/n]: "
            )
            chunk_dir = base_dir / f"chunk_{chunk_idx}"
            _save_chunk(frame_list, left_pose_list, right_pose_list, chunk_dir, camera)
            
            # while True:
            #     resp = input(prompt).strip().lower()
            #     if resp in ("y", "yes"):
            #         chunk_dir = base_dir / f"chunk_{chunk_idx}"
            #         _save_chunk(frame_list, left_pose_list, right_pose_list, chunk_dir, camera)
            #         break
            #     elif resp in ("n", "no"):
            #         print(f"Skipping save for chunk {chunk_idx}; data for this chunk will be discarded.")
            #         break
            #     else:
            #         print("Please answer 'y' or 'n'.")
    else:
        break  # single run, fall through to save block below

if not use_chunks:
    n_saved = min(len(frame_list), len(left_pose_list), len(right_pose_list))
    # Save to disk: rgbd.npz, poses, video.mp4, depth.mp4 (no PNG frames)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {n_saved} frames (rgbd + poses + videos) to {base_dir}...")

    colors = np.stack([frame_list[i]["color"] for i in range(n_saved)], axis=0)  # (N,H,W,3)
    depth_stack = None
    if camera == "azure":
        depths = [frame_list[i].get("depth") for i in range(n_saved)]
        if any(d is not None for d in depths):
            first_valid = next((d for d in depths if d is not None), None)
            if first_valid is not None:
                H_d, W_d = first_valid.shape
                depth_stack = np.zeros((n_saved, H_d, W_d), dtype=first_valid.dtype)
                for i, d in enumerate(depths):
                    if d is not None:
                        depth_stack[i] = d

    if depth_stack is not None:
        np.savez(base_dir / "rgbd.npz", color=colors, depth=depth_stack)
    else:
        np.savez(base_dir / "rgbd.npz", color=colors)

    np.savez(base_dir / "left_arm_poses.npz", *left_pose_list[:n_saved])
    np.savez(base_dir / "right_arm_poses.npz", *right_pose_list[:n_saved])

    H, W, _ = frame_list[0]["color"].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(base_dir / "video.mp4"), fourcc, 30.0, (W, H))
    for i in range(n_saved):
        video_writer.write(frame_list[i]["color"])
    video_writer.release()

    if depth_stack is not None:
        depth_writer = cv2.VideoWriter(str(base_dir / "depth.mp4"), fourcc, 30.0, (W, H))
        max_depth_mm = 1500.0
        for i in range(n_saved):
            d = depth_stack[i].astype(np.float32)
            d_norm = np.clip(d / max_depth_mm, 0.0, 1.0)
            d_img = (d_norm * 255.0).astype(np.uint8)
            depth_writer.write(cv2.applyColorMap(d_img, cv2.COLORMAP_JET))
        depth_writer.release()

    print(
        f"Saved rgbd.npz to {base_dir}: "
        f"color shape={colors.shape}, "
        f"depth shape={None if depth_stack is None else depth_stack.shape}, "
        f"left_poses={len(left_pose_list[:n_saved])}, right_poses={len(right_pose_list[:n_saved])}"
    )

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done.")

print("Return to home and shutdown")
# left_arm.home()
# right_arm.home()
left_arm.shutdown()
right_arm.shutdown()
