"""Moves the robot arm in a figure eight pattern and captures images of the AprilTag."""

# %%
import numpy as np

from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import pyzed.sl as sl
import cv2

left_arm = Robot(namespace="")
left_arm.wait_until_ready()

# %%
print(left_arm.end_effector_pose)
print(left_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
homing_pose = left_arm.end_effector_pose.copy()


# %%
# Paremeters for the circle
radius = 0.2  # [m]
center = np.array([0.4, 0.4, 0.4])
ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# %%
left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    # file_path="config/control/gravity_compensation.yaml"
    # file_path="config/control/default_operational_space_controller.yaml"
    # file_path="config/control/clipped_cartesian_impedance.yaml"
    file_path="config/control/default_cartesian_impedance.yaml"
)

# %%
# The set_target will directly publish the pose to /target_pose
ee_poses = []
target_poses = []
ts = []

print("Starting to draw a circle...")
t = 0.0
target_pose = left_arm.end_effector_pose.copy()
print("taget_pose_rotation:", target_pose.orientation)
target_pose.orientation = Rotation.from_matrix(np.array([[0.0, 1.0, 0.0],
                                                          [1.0, 0.0, 0.0],
                                                          [0.0, 0.0, -1.0]]))
center = np.array([0.0, 0.4, 0.4])
target_pose.position = np.array([0.0, 0.4, 0.4])
print("taget_pose_rotation:", target_pose)
rate = left_arm.node.create_rate(ctrl_freq)
left_arm.move_to(pose=target_pose, speed=0.15)

# Setup zed camera
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.sdk_verbose = 1
init_params.camera_resolution = sl.RESOLUTION.AUTO
init_params.camera_fps = 30
zed.open(init_params)
image = sl.Mat()
runtime_params = sl.RuntimeParameters()

frame_count = 0
pose_count = 0
pose_list  = []

while t < max_time:
    
    if frame_count % 13 == 0:
        # Save the pose
        p = left_arm.end_effector_pose.copy()
        pose_list.append(np.array([p.position[0], p.position[1], p.position[2],
         p.orientation.as_quat()[0], p.orientation.as_quat()[1],
          p.orientation.as_quat()[2], p.orientation.as_quat()[3]]))
        
        # Take the image
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT) # can we get a combined mixed view?
            frame = image.get_data()
            cv2.imwrite(f"image_pose_{pose_count}.png", frame)
            print(f"Image Captured {pose_count}")
            # detections = apriltag_image([f"./image_pose_{pose_count}.png"], output_images=True, display_images=True)
            # print("Detections: ", detections[1])
            pose_count += 1
        input("Press Enter to continue...")
        

    frame_count += 1
    
    x = radius * np.sin(2 * np.pi * sin_freq_y * t) + center[0]
    y = center[1]
    z = radius * np.sin(2 * np.pi * sin_freq_z * t) + center[2]
    target_pose.position = np.array([x, y, z])

    left_arm.set_target(pose=target_pose)

    rate.sleep()

    ee_poses.append(left_arm.end_effector_pose.copy())
    target_poses.append(left_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq

np.savez("figure_eight_poses_1_28.npz", *pose_list)

while t < max_time + 1.0:
    # Just wait a bit for the end effector to settle

    rate.sleep()

    ee_poses.append(left_arm.end_effector_pose.copy())
    target_poses.append(left_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq


print("Done drawing a circle!")

print("Going back home.")
left_arm.home()

# %%
left_arm.shutdown()
