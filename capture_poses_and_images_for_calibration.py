"""Moves the robot arm in a figure eight pattern and captures images of the AprilTag."""

import argparse
import numpy as np
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import pyzed.sl as sl
import cv2
import time
import k4a


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
initial_rotation = np.pi/2 # rotation arounnd robot z axis (counterclockwise)

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

center = np.array([0.0, 0.4, 0.4])
target_pose.position = np.array([0.0, 0.4, 0.4])
print("taget_pose_rotation:", target_pose)
rate = left_arm.node.create_rate(ctrl_freq)
left_arm.move_to(pose=target_pose, speed=0.15)

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
pose_list  = []
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"

# main trajectory loop
while t < max_time:
    
    if frame_count % 13 == 0:
        # Wait for arm to settle
        time.sleep(2.0)
        # Save the pose
        p = left_arm.end_effector_pose.copy()
        pose_list.append(np.array([p.position[0], p.position[1], p.position[2],
            p.orientation.as_quat()[0], p.orientation.as_quat()[1],
            p.orientation.as_quat()[2], p.orientation.as_quat()[3]]))
        
        # Take the image and save it
        if camera == "zed":
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                cv2.imwrite(f"{DATAPATH}/images/calibration_image_{pose_count}.png", frame)
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
            cv2.imwrite(f"{DATAPATH}/images/calibration_image_{pose_count}.png", color_bgr)
            print(f"Image Captured {pose_count}")
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

# save all poses
np.savez(f"{DATAPATH}/poses/calibration_poses.npz", *pose_list)

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done drawing a circle!")


print("return to home and shutdown")
left_arm.home()
left_arm.shutdown()
