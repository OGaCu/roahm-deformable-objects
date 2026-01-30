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

def _load_apriltag_transforms(max_images):
    detection_transforms = []
    count = 0
    for i in range(1, max_images + 1):
        detections = apriltag_image([f"./image_pose_{i}_0128.png"], output_images=True, display_images=True, tag_size=0.09, tag_family="tag16h5")
        # detections = apriltag_image([f"./image_pose_{i}.png"], output_images=True, display_images=True)
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

        # detection_transforms.append(detections[1])
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


def _mean_se3(transforms, max_iters=20, tol=1e-9):
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

gripper2tag = np.array(    [[0, 0, -1, 0.062],
                            [0, -1, 0, 0],
                            [-1, 0, 0, -0.0175],
                            [0, 0, 0, 1]])

def main():
    max_images = 30
    detection_transforms = _load_apriltag_transforms(max_images)

    t_tag_in_cam_frame, r_tag_in_cam_frame = _split_transforms(detection_transforms)

    positions, orientations = _load_figure_eight_poses("figure_eight_poses_1_28.npz", max_images)
    # positions, orientations = _load_figure_eight_poses("figure_eight_poses.npz", max_images)

    t_base2gripper = positions[0:max_images]
    r_base2gripper = orientations[0:max_images]

    # r_gripper2cam, t_gripper2cam = _invert_rt_pairs(r_tag_in_cam_frame, t_tag_in_cam_frame)


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


    t_cam2base_list = []
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
        t_cam2base_list.append(t_cam2tag_mat @ t_tag2base_mat)

    print("LAST CAM2BASE")
    print(t_cam2base_list[-1])

    t_mean = _mean_se3(t_cam2base_list)
    print("SE3 mean cam 2 base:\n", t_mean)


if __name__ == "__main__":
    main()
