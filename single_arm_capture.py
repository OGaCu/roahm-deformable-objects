"""Moves the robot arm along a joint trajectory and streams (camera image, EE pose) pairs.

=== High-level flow ===
1. Trajectory is predefined: joint waypoints from a .task file.
   - Waypoints have no inherent "fps"; the robot interpolates over time_to_goal per segment.
2. A separate streaming thread runs at --stream-hz (default 10): each tick it captures one image and
   reads the current EE pose, then saves the pair. So you get (image, pose) pairs at the stream rate.
3. Trajectory "fps" and stream rate are independent: the trajectory is defined by waypoints and
   controller timing; the stream rate is how often we sample (image, pose). Frames and poses are
   buffered in RAM during the run; all saves to disk happen after motion and streaming finish, so
   actual capture rate can reach camera FPS (e.g. 30 fps) without disk I/O as a bottleneck.

=== Order of pose vs camera capture (per stream tick) ===
- Azure: pose_before → get_capture() [image acquired] → store pose_before, so the stored pose
  is slightly before capture time.
"""

import argparse
import json
import threading
from pathlib import Path
import time

import numpy as np
from crisp_py.robot import Robot
import cv2
import k4a


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
    azure_device,
    azure_transformation,
    pose_list: list,
    frame_list: list,
    lock: threading.Lock,
    rate_hz: float,
) -> None:
    """
    Run at target rate_hz: each tick = one (image, EE pose) pair appended to lists. No disk I/O
    during the loop so rate is limited only by camera FPS and get_capture(); saves happen after the run.
    Order per tick:
      - Azure: read EE pose (before) → get_capture() →
               depth_image_to_color_camera() → append (color_bgr, depth_aligned, pose_before).
    """
    period = 1.0 / rate_hz
    while capture_active.is_set():
        t0 = time.time()
        try:
            p_first = left_arm.end_effector_pose.copy()
            capture = azure_device.get_capture(-1)
            pose_vec_first = np.array([
                p_first.position[0], p_first.position[1], p_first.position[2],
                p_first.orientation.as_quat()[0], p_first.orientation.as_quat()[1],
                p_first.orientation.as_quat()[2], p_first.orientation.as_quat()[3],
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
        except Exception as e:
            print(f"Streaming capture error: {e}")
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


def _apply_frame_pose_delay(
    frame_list: list,
    pose_list: list,
    delay: int,
) -> tuple[list, list]:
    if delay <= 0:
        return frame_list, pose_list

    if len(frame_list) <= delay:
        frame_list = []
    else:
        frame_list = frame_list[delay:]

    if len(pose_list) <= delay:
        pose_list = []
    else:
        pose_list = pose_list[:-delay]

    return frame_list, pose_list


def _save_run(
    frame_list: list,
    pose_list: list,
    output_dir: Path,
    video_fps: float,
    max_depth_mm: float,
) -> int:
    n_saved = min(len(frame_list), len(pose_list))
    if n_saved <= 0:
        print(f"No data to save for {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    colors = np.stack([frame_list[i]["color"] for i in range(n_saved)], axis=0)  # (N,H,W,3)
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

    rgbd_path = output_dir / "rgbd.npz"
    if depth_stack is not None:
        np.savez(rgbd_path, color=colors, depth=depth_stack)
    else:
        np.savez(rgbd_path, color=colors)

    np.savez(output_dir / "single_arm_poses.npz", *pose_list[:n_saved])

    H, W, _ = frame_list[0]["color"].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_dir / "video.mp4"), fourcc, video_fps, (W, H))
    for i in range(n_saved):
        video_writer.write(frame_list[i]["color"])
    video_writer.release()

    if depth_stack is not None:
        depth_writer = cv2.VideoWriter(str(output_dir / "depth.mp4"), fourcc, video_fps, (W, H))
        for i in range(n_saved):
            d = depth_stack[i].astype(np.float32)
            d_norm = np.clip(d / max_depth_mm, 0.0, 1.0)
            d_img = (d_norm * 255.0).astype(np.uint8)
            d_color = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
            depth_writer.write(d_color)
        depth_writer.release()

    print(
        f"Saved rgbd.npz to {output_dir}: "
        f"color shape={colors.shape}, "
        f"depth shape={None if depth_stack is None else depth_stack.shape}, "
        f"poses={len(pose_list[:n_saved])}"
    )
    return n_saved


def main() -> None:
    DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"
    DEFAULT_TASK = f"{DATAPATH}/right_traj.task"
    STREAM_HZ_DEFAULT = 30.0
    TIME_TO_GOAL = 3.0  # seconds per joint waypoint when using joint trajectory
    SETTLE_SECONDS = 0.5
    POST_SAVE_SETTLE_SECONDS = 1.0
    VIDEO_FPS = 30.0
    MAX_DEPTH_MM = 1500.0
    CAMERA_COLOR_FORMAT = k4a.EImageFormat.COLOR_BGRA32
    CAMERA_COLOR_RESOLUTION = k4a.EColorResolution.RES_720P
    CAMERA_DEPTH_MODE = k4a.EDepthMode.NFOV_UNBINNED
    CAMERA_FPS = k4a.EFramesPerSecond.FPS_30
    # Delay between the captured robot poses and the captured images which we have noticed exists
    # we assume this is because the camera capture does not block this script and therefore creates a delay
    IMAGE_POSE_FRAME_DELAY = 7

    parser = argparse.ArgumentParser(description="Capture poses and images with Azure Kinect.")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Path to .task file for joint waypoints (default: DATAPATH/right_traj.task).",
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
        default=STREAM_HZ_DEFAULT,
        help="Target streaming rate in Hz (image + EE pose per tick). For 30 fps use 30 and ensure Azure camera_fps=FPS_30; actual rate limited by camera, get_capture, and disk I/O.",
    )
    parser.add_argument(
        "--waypoints-per-chunk",
        type=int,
        default=None,
        help="Save data every N waypoints, clear RAM, then continue. Output: chunk_0/, chunk_1/, ... "
        "(rgbd.npz, poses, video.mp4, depth.mp4 per chunk; no PNG frames). None = single run (default).",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=None,
        help="Optional upper limit on number of waypoints to execute.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index of first waypoint in the .task to use for capture.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="0-based one-past-last waypoint index to use from the .task (default: full length).",
    )
    args = parser.parse_args()
    task_path = args.task or DEFAULT_TASK
    seq_name = args.seq_name
    waypoints_per_chunk = args.waypoints_per_chunk
    max_waypoints = args.max_waypoints
    start_index = args.start_index
    end_index = args.end_index
    base_dir = Path(DATAPATH) / "captured_data_single_arm" / seq_name

    # initialize robot
    left_arm = Robot(namespace="")
    left_arm.wait_until_ready()

    print("Going to home position...")
    left_arm.home()

    # Setup Params and waypoints
    joint_waypoints = parse_task_joints(task_path)
    if not joint_waypoints:
        raise RuntimeError(f"No joint waypoints found in {task_path}")
    total_joint_waypoints = len(joint_waypoints)
    if start_index < 0:
        start_index = 0
    if end_index is None or end_index > total_joint_waypoints:
        end_index = total_joint_waypoints
    if start_index >= end_index:
        raise ValueError(
            f"Invalid waypoint segment: start-index={start_index}, end-index={end_index}, "
            f"available range is [0, {total_joint_waypoints})."
        )
    joint_waypoints = joint_waypoints[start_index:end_index]
    n_joint_waypoints = len(joint_waypoints)
    if max_waypoints is not None and max_waypoints > 0 and max_waypoints < n_joint_waypoints:
        print(f"Limiting joint waypoints in segment from {n_joint_waypoints} to max-waypoints={max_waypoints}.")
        n_joint_waypoints = max_waypoints
        joint_waypoints = joint_waypoints[:n_joint_waypoints]
    print(
        f"Joint mode: using waypoint segment [{start_index}, {end_index}) from .task "
        f"({n_joint_waypoints} waypoints)."
    )
    left_arm.controller_switcher_client.switch_controller("joint_trajectory_controller")

    print("Starting capture...")

    # Setup camera
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
        color_format=CAMERA_COLOR_FORMAT,
        color_resolution=CAMERA_COLOR_RESOLUTION,
        # Depth: e.g. OFF, NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR
        depth_mode=CAMERA_DEPTH_MODE,
        # FPS: 5, 15, or 30. This caps max frames you can get; stream_hz should be <= this.
        camera_fps=CAMERA_FPS,
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
    frame_list = []  # each item: {"color": bgr_array, "depth": array or None}
    pose_list_lock = threading.Lock()
    capture_active = threading.Event()
    stream_hz = args.stream_hz

    use_chunks = waypoints_per_chunk is not None and waypoints_per_chunk > 0
    if use_chunks:
        chunk_size = waypoints_per_chunk
        num_chunks = (n_joint_waypoints + chunk_size - 1) // chunk_size
        print(f"Chunked joint mode: {n_joint_waypoints} waypoints in {num_chunks} chunks of up to {chunk_size}.")

    for chunk_idx in range(num_chunks if use_chunks else 1):
        if use_chunks:
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_joint_waypoints)
            frame_list.clear()
            pose_list.clear()
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: waypoints {chunk_start}..{chunk_end - 1}")
        else:
            chunk_start = 0
            chunk_end = n_joint_waypoints

        capture_active.set()
        stream_thread = threading.Thread(
            target=_streaming_capture_loop,
            kwargs=dict(
                capture_active=capture_active,
                left_arm=left_arm,
                azure_device=device,
                azure_transformation=azure_transformation,
                pose_list=pose_list,
                frame_list=frame_list,
                lock=pose_list_lock,
                rate_hz=stream_hz,
            ),
            daemon=True,
        )
        stream_thread.start()

        joint_names = left_arm.config.joint_names
        for i in range(chunk_start, chunk_end):
            left_arm.joint_trajectory_controller_client.send_joint_config(
                joint_names, joint_waypoints[i].tolist(), time_to_goal=TIME_TO_GOAL, blocking=False
            )
            time.sleep(TIME_TO_GOAL)
            print(f"Joint waypoint {i + 1}/{n_joint_waypoints} sent")
        time.sleep(SETTLE_SECONDS)  # allow last motion to settle
        capture_active.clear()
        stream_thread.join(timeout=2.0)
        print(f"Streaming capture finished: {len(pose_list)} camera–EE pairs (buffered in RAM)")

        # Realign frames and poses due to camera delay.
        frame_list, pose_list = _apply_frame_pose_delay(
            frame_list, pose_list, IMAGE_POSE_FRAME_DELAY
        )
        n_saved = min(len(frame_list), len(pose_list))
        print(f"Chunk/run finished: {n_saved} frames, {len(pose_list)} poses")

        if use_chunks:
            if n_saved > 0:
                chunk_dir = base_dir / f"chunk_{chunk_idx}"
                _save_run(frame_list, pose_list, chunk_dir, VIDEO_FPS, MAX_DEPTH_MM)
        else:
            _save_run(frame_list, pose_list, base_dir, VIDEO_FPS, MAX_DEPTH_MM)
            break

    print("Waiting for robot to settle...")
    time.sleep(POST_SAVE_SETTLE_SECONDS)
    print("Done drawing a circle!")

    print("return to home and shutdown")
    left_arm.home()
    left_arm.shutdown()


if __name__ == "__main__":
    main()
