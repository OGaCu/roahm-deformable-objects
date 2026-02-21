import argparse

from apriltag_image import apriltag_image
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def _skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])

def _se3_log(T, eps=1e-9):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    w = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(w)
    W = _skew(w)
    if theta < eps:
        V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V_inv = np.eye(3) - 0.5 * W + (1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)) * (W @ W)
    v = V_inv @ p
    return np.hstack([w, v])

def _se3_exp(xi, eps=1e-9):
    w = xi[0:3]
    v = xi[3:6]
    theta = np.linalg.norm(w)
    W = _skew(w)
    if theta < eps:
        R = np.eye(3) + W + 0.5 * (W @ W)
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        C = (1.0 - A) / (theta * theta)
        R = np.eye(3) + A * W + B * (W @ W)
        V = np.eye(3) + B * W + C * (W @ W)
    p = V @ v
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T

def _load_apriltag_transforms(max_images, DATAPATH, camera="azure"):
    detection_transforms = []
    count = 0
    for i in range(max_images):
        detections = apriltag_image([f"{DATAPATH}/images/calibration_image_{i}.png"], output_images=True, display_images=True, tag_size=0.095, tag_family="tag36h11", camera=camera)
        if detections is None or len(detections) == 0:
            print("no detection for image ", i)
            detection_transforms.append(None)
            count += 1
            continue
        expected_tag_id = 3
        for i in range(0, len(detections), 4):
            found = False
            if(detections[i].tag_id == expected_tag_id):
                if found:
                    assert(False) #Means detected 2 tag_id 3 tags in image
                detection_transforms.append(detections[i+1])
                found = True

    print(f"Total no detection images: {count} out of {max_images}")
    return detection_transforms


def _split_transforms(transforms):
    t_list = []
    r_list = []
    for transform in transforms:
        if transform is None:
            t_list.append(None)
            r_list.append(None)
        else:
            t_list.append(transform[0:3, 3])
            r_list.append(transform[0:3, 0:3])
    return t_list, r_list


def _load_figure_eight_poses(npz_path, max_images):
    np_info = np.load(npz_path)
    positions = []
    orientations = []
    for i in range(max_images):
        np_stuff = np_info["arr_" + str(i)]
        positions.append(np_stuff[0:3])
        orientations.append(Rotation.from_quat(np_stuff[3:]).as_matrix())
    return positions, orientations


def _invert_rt_pairs(r_list, t_list):
    r_out, t_out = [], []
    for r, t in zip(r_list, t_list):
        r_inv = r.T
        t_inv = -r_inv @ t
        r_out.append(r_inv)
        t_out.append(t_inv)
    return r_out, t_out


def _mean_se3(transforms, max_iters=100, tol=1e-9):
    t_mean = transforms[0]
    for _ in range(max_iters):
        xi_sum = np.zeros(6)
        for t_i in transforms:
            xi_sum += _se3_log(np.linalg.inv(t_mean) @ t_i)
        xi_avg = xi_sum / len(transforms)
        if np.linalg.norm(xi_avg) < tol:
            break
        t_mean = t_mean @ _se3_exp(xi_avg)
    return t_mean

gripper2tag = np.array(    [[0, 0, -1, 0.02],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0.088],
                            [0, 0, 0, 1]])
# gripper2tag = np.array(    [[0, 0, -1, 0],
#                             [0, -1, 0, 0],
#                             [-1, 0, 0, 0],
#                             [0, 0, 0, 1]])
# gripper2tag_2 = np.array(    [[1, 0, 0, -0.08],
#                             [0, 1, 0, 0],
#                             [0, 0, 1, -0.02],
#                             [0, 0, 0, 1]])
# print("gripperthing: ", gripper2tag @ gripper2tag_2)

def main():
    parser = argparse.ArgumentParser(description="Calculate base-to-camera transform from poses and AprilTag detections.")
    parser.add_argument(
        "--camera",
        type=str,
        choices=["azure", "zed"],
        default="azure",
        help="Camera used for capture: 'azure' or 'zed' (default: azure)",
    )
    args = parser.parse_args()
    camera = args.camera

    max_images = 58
    DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"

    detection_transforms = _load_apriltag_transforms(max_images, DATAPATH, camera=camera)

    t_tag_in_cam_frame, r_tag_in_cam_frame = _split_transforms(detection_transforms)

    positions, orientations = _load_figure_eight_poses(f"{DATAPATH}/poses/calibration_poses.npz", max_images)

    t_base2gripper = positions[0:max_images]
    r_base2gripper = orientations[0:max_images]



    r_base2tag, t_base2tag = [], []
    for r, t in zip(r_base2gripper, t_base2gripper):
        t_base2gripper_mat = np.eye(4)
        t_base2gripper_mat[0:3, 0:3] = r
        t_base2gripper_mat[0:3, 3] = t

        t_base2tag_mat = t_base2gripper_mat @ gripper2tag
        r_base2tag.append(t_base2tag_mat[0:3, 0:3])
        t_base2tag.append(t_base2tag_mat[0:3, 3])

    r_tag2base, t_tag2base = [], []
    for r, t in zip(r_base2tag, t_base2tag):
        t_base2tag_mat = np.eye(4)
        t_base2tag_mat[0:3, 0:3] = r
        t_base2tag_mat[0:3, 3] = t
        t_tag2base_mat = np.linalg.inv(t_base2tag_mat)
        r_b2t = t_tag2base_mat[0:3, 0:3]
        t_b2t = t_tag2base_mat[0:3, 3]
        r_tag2base.append(r_b2t)
        t_tag2base.append(t_b2t)


    t_base2cam_list = []
    for index in range(max_images):
        if r_tag_in_cam_frame[index] is None or t_tag_in_cam_frame[index] is None:
            print(f"Skipping image {index+1} - no detection")
            continue
        t_tag2base_mat = np.eye(4)
        t_tag2base_mat[0:3, 0:3] = r_tag2base[index]
        t_tag2base_mat[0:3, 3] = t_tag2base[index]
        t_cam2tag_mat = np.eye(4)
        t_cam2tag_mat[0:3, 0:3] = r_tag_in_cam_frame[index]
        t_cam2tag_mat[0:3, 3] = t_tag_in_cam_frame[index]
        t_base2cam_list.append(t_cam2tag_mat @ t_tag2base_mat)

    base2cam_mean = _mean_se3(t_base2cam_list)
    print("SE3 mean base 2 cam:\n", base2cam_mean)
    np.savez(f"{DATAPATH}/poses/base2cam_transform.npz", base2cam_mean)


if __name__ == "__main__":
    main()
