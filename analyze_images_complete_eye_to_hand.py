from apriltag_image import apriltag_image
import cv2
import numpy as np
from scipy.spatial.transform import Rotation 

detection_transforms = []
max_images = 30
for i in range(1, max_images+1):
    detections = apriltag_image([f"./image_pose_{i}.png"], output_images=True, display_images=True)
    detection_transforms.append(detections[1])
print(detection_transforms)
t_cam2gripper = []
r_cam2gripper = []

for detection in detection_transforms:
    t_cam2gripper.append(detection[0:3, 3])
    r_cam2gripper.append(detection[0:3, 0:3])
print(t_cam2gripper)
print(r_cam2gripper)

### Base 2 Gripper transforms
# achieved_pos1 = np.array([0.56, -0.21, 0.44])
# achieved_ori1 = np.array([[-0.39569914, 0.04180925, 0.91742802],
#                           [-0.69671214, 0.63717833, -0.32953904],
#                           [-0.59834303, -0.76958155, -0.22300191]])

# achieved_pos2 = np.array([0.51, -0.26, 0.5])
# achieved_ori2 = np.array([[-0.25095132, -0.38018741, 0.890214],
#                           [-0.8668955, -0.32093063, -0.38143902],
#                           [0.43071526, -0.86744514, -0.24904479]])

# achieved_pos3 = np.array([0.53, -0.15, 0.55])
# achieved_ori3 = np.array([[0.11042751, -0.29415084, 0.94935823],
#                           [-0.48716977, -0.84859911, -0.20626481],
#                           [0.86629751, -0.43972132, -0.23701008]])

# achieved_pos4 = np.array([0.55, -0.01, 0.59])
# achieved_ori4 = np.array([[0.12008366, -0.02115112, 0.99253844],
#                           [-0.02115112, -0.99960059, -0.01874262],
#                           [0.99253844, -0.01874262, -0.12048306]])

# achieved_pos5 = np.array([0.44, 0.01, 0.66])
# achieved_ori5 = np.array([[0.02715695, -0.11099294, 0.99345008],
#                           [-0.05151567, -0.99265144, -0.10949548],
#                           [0.99830288, -0.04820468, -0.03267526]])

# achieved_pos6 = np.array([0.45, 0.08, 0.57])
# achieved_ori6 = np.array([[0.12073953, -0.07136262, 0.99011582],
#                           [0.77001991, -0.62274256, -0.13878416],
#                           [0.62649126, 0.77916563, -0.020239]])

# achieved_pos7 = np.array([0.51, 0.07, 0.51])
# achieved_ori7 = np.array([[0.15814902, 0.04933581, 0.98618196],
#                           [0.9856845, 0.05122055, -0.16063166],
#                           [-0.05843768, 0.99746801, -0.04052907]])

# achieved_pos8 = np.array([0.56, -0.04, 0.43])
# achieved_ori8 = np.array([[0.1601518, 0.37584872, 0.91273717],
#                           [0.7695146, 0.53158963, -0.35392054],
#                           [-0.6182222, 0.75904559, -0.20408604]])

np_info = np.load("figure_eight_poses.npz")
positions = []
orientations = []
for i in range(max_images):
    np_stuff = np_info["arr_"+str(i)]
    positions.append(np_stuff[0:3])
    orientations.append(Rotation.from_quat(np_stuff[3:]).as_matrix())

print(positions)
print(orientations)
# t_base2gripper = [achieved_pos1, achieved_pos2, achieved_pos3, achieved_pos4, achieved_pos5, achieved_pos6, achieved_pos7, achieved_pos8]
# r_base2gripper = [achieved_ori1, achieved_ori2, achieved_ori3, achieved_ori4, achieved_ori5, achieved_ori6, achieved_ori7, achieved_ori8]

t_base2gripper = positions
r_base2gripper = orientations

r_gripper2cam, t_gripper2cam = [], []
for R, t in zip(r_cam2gripper, t_cam2gripper):
    R_b2g = R.T
    t_b2g = -R_b2g @ t
    r_gripper2cam.append(R_b2g)
    t_gripper2cam.append(t_b2g)

gripper2tag = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0.062], 
                        [0, 0, -1, 0.0175], 
                        [0, 0, 0, 1]])
# gripper2tag = np.array([[0, 1, 0, 0],
#                         [1, 0, 0, 0], 
#                         [0, 0, -1, 0], 
#                         [0, 0, 0, 1]])


r_base2tag, t_base2tag = [], []
for r, t in zip(r_base2gripper, t_base2gripper):
    T_base2gripper = np.eye(4)
    T_base2gripper[0:3, 0:3] = r
    T_base2gripper[0:3, 3] = t


    T_gripper2tag = T_base2gripper @ gripper2tag
    # print(T_gripper2tag)
    r_base2tag.append(T_gripper2tag[0:3, 0:3])
    t_base2tag.append(T_gripper2tag[0:3, 3])

r_tag2base, t_tag2base = [], []
for R, t in zip(r_base2tag, t_base2tag):
    R_b2g = R.T
    t_b2g = -R_b2g @ t
    r_tag2base.append(R_b2g)
    t_tag2base.append(t_b2g)


R, t = cv2.calibrateHandEye(
        R_gripper2base=r_tag2base,
        t_gripper2base=t_tag2base,
        R_target2cam=r_cam2gripper, # cam to tag
        t_target2cam=t_cam2gripper)
print("new eye to hand calibration")
print("R", R)
print("t", t)
print("end")