"""Moves the robot arm in a figure eight pattern and captures images of the AprilTag."""

import argparse
import json
from pathlib import Path

import numpy as np
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import pyzed.sl as sl
import cv2
import time
import k4a
import time



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


parser = argparse.ArgumentParser(description="Capture poses and images with specified camera.")
parser.add_argument(
    "--camera",
    type=str,
    choices=["azure", "zed"],
    default="azure",
    help="Camera to use: 'azure' or 'zed' (default: zed)",
)
args = parser.parse_args()
camera = args.camera
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"


# initialize robot
left_arm = Robot(namespace="left")
right_arm = Robot(namespace="right")

left_arm.wait_until_ready()
right_arm.wait_until_ready()

print("Going to home position...")
left_arm.home()
right_arm.home()

# Setup Params

left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/clipped_cartesian_impedance.yaml"
)

left_arm.cartesian_controller_parameters_client.set_parameters([
    ("task.error_clip.x", 0.005), ("task.error_clip.y", 0.005), ("task.error_clip.z", 0.005),
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

# waypoint list

left_waypoints = parse_task_poses(f"{DATAPATH}/left_traj.task")
right_waypoints = parse_task_poses(f"{DATAPATH}/right_traj.task")

# set initial target pose and orientation
print("Starting to capture...")
left_target_pose = left_arm.end_effector_pose.copy()
right_target_pose = right_arm.end_effector_pose.copy()

# Setup zed camera
if camera == "zed":
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
        depth_mode=k4a.EDepthMode.WFOV_2X2BINNED,
        camera_fps=k4a.EFramesPerSecond.FPS_15,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False)
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        exit(-1)


# data capture variables
frame_count = 0
pose_count = 0
left_pose_list  = []
right_pose_list  = []

left_target_pose.position = left_waypoints[0][0:3,3]
left_target_pose.orientation = Rotation.from_matrix(left_waypoints[0][0:3, 0:3])
right_target_pose.position = right_waypoints[0][0:3,3]
right_target_pose.orientation = Rotation.from_matrix(right_waypoints[0][0:3, 0:3])
left_waypoint_index = 0
right_waypoint_index = 0

# print("taget_pose_rotation:", target_pose)
left_arm.move_to(pose=left_target_pose, speed=0.15)
right_arm.move_to(pose=right_target_pose, speed=0.15)

# assert(len(left_waypoints) == len(right_waypoints))

# main trajectory loop
start_time = 0.0
while left_waypoint_index < len(left_waypoints) and right_waypoint_index < len(right_waypoints):
    end_time = time.time()
    print(f"Time taken: {1/(end_time - start_time)} Hz")
    start_time = time.time()

    if frame_count % 1 == 0:
        # Save the pose
        p_left = left_arm.end_effector_pose.copy()
        left_pose_list.append(np.array([p_left.position[0], p_left.position[1], p_left.position[2],
            p_left.orientation.as_quat()[0], p_left.orientation.as_quat()[1],
            p_left.orientation.as_quat()[2], p_left.orientation.as_quat()[3]]))
        p_right = right_arm.end_effector_pose.copy()
        right_pose_list.append(np.array([p_right.position[0], p_right.position[1], p_right.position[2],
            p_right.orientation.as_quat()[0], p_right.orientation.as_quat()[1],
            p_right.orientation.as_quat()[2], p_right.orientation.as_quat()[3]]))
        # Take the image and save it
        if camera == "zed":
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                cv2.imwrite(f"{DATAPATH}/images/image_pose_{pose_count}.png", frame)
                print(f"Image Captured {pose_count}")
                pose_count += 1
            else:
                print(f"ERROR: Failed to capture image {pose_count}")
                pose_count += 1
        else:
            capture = device.get_capture(-1)
            color_image = capture.color
            color_image_data = color_image.data  # NumPy array (BGRA)
            color_bgr = cv2.cvtColor(color_image_data, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(f"{DATAPATH}/images/double_arm_image_{pose_count}.png", color_bgr)
            # Save full RGB-D as npz (color BGR, depth raw uint16 in mm)
            depth_image = capture.depth
            save_kw = {"color": color_bgr}
            if depth_image is not None:
                save_kw["depth"] = depth_image.data
            np.savez(f"{DATAPATH}/images/double_arm_rgbd_{pose_count}.npz", **save_kw)
            print(f"Image Captured {pose_count}")
            pose_count += 1
            if status != k4a.EStatus.SUCCEEDED:
                exit(-1)
        

    frame_count += 1
    print(f"Current position: {left_arm.end_effector_pose.position}")
    print(f"Target position: {left_target_pose.position}")
    print(f"Difference: {sum(left_arm.end_effector_pose.position - left_target_pose.position)}")
    if abs(sum(left_arm.end_effector_pose.position - left_target_pose.position)) < 0.05:
        left_waypoint_index += 1
        if(left_waypoint_index < len(left_waypoints)):
            left_target_pose.position = left_waypoints[left_waypoint_index][0:3,3]
            print(f"Waypoint {left_waypoint_index} reached")
            left_target_pose.orientation = Rotation.from_matrix(left_waypoints[left_waypoint_index][0:3, 0:3])

    if abs(sum(right_arm.end_effector_pose.position - right_target_pose.position)) < 0.05:
        right_waypoint_index += 1
        if(right_waypoint_index < len(right_waypoints)):
            right_target_pose.position = right_waypoints[right_waypoint_index][0:3,3]
            print(f"Waypoint {right_waypoint_index} reached")
            right_target_pose.orientation = Rotation.from_matrix(right_waypoints[right_waypoint_index][0:3, 0:3])

    # send target to controller
    left_arm.set_target(pose=left_target_pose)
    right_arm.set_target(pose=right_target_pose)

assert(len(left_pose_list) == len(right_pose_list))
# save all poses
np.savez(f"{DATAPATH}/poses/left_arm_poses.npz", *left_pose_list)
np.savez(f"{DATAPATH}/poses/right_arm_poses.npz", *right_pose_list)

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done drawing a circle!")


print("return to home and shutdown")
left_arm.home()
right_arm.home()
left_arm.shutdown()
right_arm.shutdown()
