"""Moves the robot arm in a figure eight pattern and captures images of the AprilTag (left arm calibration)."""

import argparse
from pathlib import Path

import cv2
import k4a
import numpy as np
import pyzed.sl as sl
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import time


parser = argparse.ArgumentParser(description="Capture poses and images for calibration with specified camera (left arm).")
parser.add_argument(
    "--camera",
    type=str,
    choices=["azure", "zed"],
    default="azure",
    help="Camera to use: 'azure' or 'zed' (default: azure)",
)
parser.add_argument(
    "--seq-name",
    type=str,
    required=True,
    help="Calibration sequence name (data saved under captured_calibration_data/{seq_name}).",
)
args = parser.parse_args()
camera = args.camera
seq_name = args.seq_name

# initialize robot
left_arm = Robot(namespace="")
left_arm.wait_until_ready()

print("Going to home position...")
left_arm.home()

# figure eight parameters
radius = 0.2  # [m]
ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# Setup Params
# this is the roataion of the end effector in the base frame ie rotation around z axis
# to move the initial positoin of the end effector change center variable lower in this file
initial_rotation = -1 *np.pi/2 # rotation arounnd robot z axis (counterclockwise)

left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/default_cartesian_impedance.yaml"
)

# set initial target pose and orientation
print("Starting to draw a circle...")
t = 0.0
target_pose = left_arm.end_effector_pose.copy()
print("taget_pose_rotation:", target_pose.orientation)
original_rotation = np.array([  [1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0]])
turn_rotation = np.array([  [np.cos(initial_rotation), -1 * np.sin(initial_rotation), 0.0],
                                [np.sin(initial_rotation), np.cos(initial_rotation), 0.0],
                                [0.0, 0.0, 1.0]])
target_pose.orientation = Rotation.from_matrix(turn_rotation @ original_rotation)

center = np.array([0.2, -0.4, 0.4])
target_pose.position = center
# print("taget_pose_rotation:", target_pose)
rate = left_arm.node.create_rate(ctrl_freq)
left_arm.move_to(pose=target_pose, speed=0.15)

# Setup camera (ZED has no depth in this script; Azure uses color-aligned depth via k4a.Transformation)
if camera == "zed":
    azure_transformation = None
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30
    zed.open(init_params)
    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()
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
        disable_streaming_indicator=False)
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        exit(-1)
    cal = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    azure_transformation = k4a.Transformation.create(cal)


# data capture variables and output layout (same pattern as single_arm_capture: frame_list of {color, depth})
frame_count = 0
pose_count = 0
pose_list = []
frame_list = []  # each item: {"color": bgr_array, "depth": color-aligned depth or None}
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"
base_dir = Path(DATAPATH) / "captured_calibration_data" / seq_name
frames_dir = base_dir / "frames"
base_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

# main trajectory loop
while t < max_time:
    
    if frame_count % 7 == 0:
        # Wait for arm to settle
        time.sleep(2.0)
        # Save the pose
        p = left_arm.end_effector_pose.copy()
        pose_list.append(np.array([p.position[0], p.position[1], p.position[2],
            p.orientation.as_quat()[0], p.orientation.as_quat()[1],
            p.orientation.as_quat()[2], p.orientation.as_quat()[3]]))
        
        # Take the image and save it (left arm); buffer color + color-aligned depth like single_arm_capture
        if camera == "zed":
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data().copy()
                cv2.imwrite(str(frames_dir / f"calibration_left_image_{pose_count}.png"), frame)
                frame_list.append({"color": frame, "depth": None})
                print(f"Left image Captured {pose_count}")
                pose_count += 1
            else:
                print(f"ERROR: Failed to capture image {pose_count}")
                pose_count += 1
        else:
            capture = device.get_capture(-1)
            color_image = capture.color
            color_image_data = color_image.data  # NumPy array (BGRA)
            color_bgr = cv2.cvtColor(color_image_data, cv2.COLOR_BGRA2BGR).copy()
            if capture.depth is not None and azure_transformation is not None:
                depth_color_img = azure_transformation.depth_image_to_color_camera(capture.depth)
                depth_data = depth_color_img.data.copy()  # (H, W) uint16, same H,W as color
            else:
                depth_data = None
            cv2.imwrite(str(frames_dir / f"calibration_left_image_{pose_count}.png"), color_bgr)
            frame_list.append({"color": color_bgr, "depth": depth_data})
            print(f"Left image Captured {pose_count}")
            pose_count += 1
            if status != k4a.EStatus.SUCCEEDED:
                exit(-1)


    frame_count += 1
    
    # compute figure-eight trajectory position
    x = radius * np.sin(2 * np.pi * sin_freq_y * t) + center[0]
    y = center[1]
    z = radius * np.sin(2 * np.pi * sin_freq_z * t) + center[2]
    target_pose.position = np.array([x, y, z])

    # send target to controller
    left_arm.set_target(pose=target_pose)
    rate.sleep()

    t += 1.0 / ctrl_freq

# save all poses for this left-arm calibration sequence
np.savez(base_dir / "left_calibration_poses.npz", *pose_list)

# save RGB-D in same format as single_arm_capture: rgbd.npz with color (N,H,W,3), depth (N,H,W) if Azure
n_saved = len(frame_list)
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
            np.savez(base_dir / "left_calibration_rgbd.npz", color=colors, depth=depth_stack)
            print(f"Saved left_calibration_rgbd.npz to {base_dir} with depth")
        else:
            np.savez(base_dir / "left_calibration_rgbd.npz", color=colors)
    else:
        np.savez(base_dir / "left_calibration_rgbd.npz", color=colors)
else:
    np.savez(base_dir / "left_calibration_rgbd.npz", color=colors)

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done drawing a circle!")


print("return to home and shutdown")
left_arm.home()
left_arm.shutdown()
