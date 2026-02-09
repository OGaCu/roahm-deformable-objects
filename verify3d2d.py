import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from analyze_images_complete_eye_to_hand import gripper2tag

# Tag positions and rotations
def load_saved_transforms(transform_file:str, num_transforms=30):
    np_info = np.load(transform_file)
    translations = [] # tag in the base frame
    rotations = [] # rotations in the base frame
    for i in range(num_transforms):
        np_stuff = np_info["arr_"+str(i)]
        translations.append(np_stuff[0:3])
        rotations.append(Rotation.from_quat(np_stuff[3:]).as_matrix())
    return translations, rotations

def get_center_tag_transforms(t_base2gripper, r_base2gripper):
    t_base2tag = []
    r_base2tag = []
    for r, t in zip(r_base2gripper, t_base2gripper):
        t_base2gripper_mat = np.eye(4)
        t_base2gripper_mat[0:3, 0:3] = r
        t_base2gripper_mat[0:3, 3] = t

        t_base2tag_mat = t_base2gripper_mat @ gripper2tag
        r_base2tag.append(t_base2tag_mat[0:3, 0:3])
        t_base2tag.append(t_base2tag_mat[0:3, 3])
    return t_base2tag, r_base2tag


# From prior we know the camera is roughly looking in the negative x axis of the base frame
def ground_truth_3d_plot(t_base2tag):
    """
    Plots the 3d positions of the tag locations
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot tag positions
    xs = [t[0] for t in t_base2tag]
    ys = [t[1] for t in t_base2tag]
    zs = [t[2] for t in t_base2tag]
    ax.scatter(xs, ys, zs, c='r', marker='o')

    all_x = xs 
    all_y = ys
    all_z = zs 
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_min + x_max) / 2.0
    mid_y = (y_min + y_max) / 2.0
    mid_z = (z_min + z_max) / 2.0
    ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
    ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
    ax.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def project_3d_to_2d(point_3d, K, T_base2cam, use_cv2=False):
    """
    Project a 3D point from base frame to 2D image coordinates.
    T_base2cam is a transformation matrix of the camera frame in the base frame coordinates
    """
    # Transform point to camera frame
    if not use_cv2:
        point_base_homog = np.hstack([point_3d, 1.0])[[-0.9980, -0.0626, -0.0035, 0.3540],
 [-0.0003, 0.0601, -0.9982, 0.1650],
 [0.0627, -0.9962, -0.0600, 1.7671],
 [0.0000, 0.0000, 0.0000, 1.0000]]
def plot_2d_points(points_2d):
    colors = np.linspace(0.0, 1.0, len(points_2d))
    plt.figure()
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, cmap='viridis')
    plt.colorbar(label='iteration')
    plt.xlabel('u (px)')
    plt.ylabel('v (px)')
    plt.title('Projected 2D points by iteration')
    plt.gca().invert_yaxis()
    plt.show()

def get_image_points(max_images, t_base2tag, K, T_base2cam):
    points_2d = []
    for i in range(max_images):
        result = project_3d_to_2d(t_base2tag[i], K, T_base2cam, use_cv2=True)
        if result is not None:
            points_2d.append(result)
    if len(points_2d) == 0:
        print("No valid projections!")
        exit()
    points_2d = np.array([p.reshape(2) for p in points_2d])
    return points_2d

def project_2d_points_on_images(max_images, points_2d):
    valid_projections = 0
    for i in range(1, max_images + 1):
        img_path = f"/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/images/image_pose_{i}.png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        # Project tag center
        if i - 1 >= len(points_2d):
            print(f"Warning: No projection available for image {i}")
            continue
        point_2d = points_2d[i - 1]
        overlay = img.copy()
        h, w = img.shape[:2]
        print(f"Image {i}:")

        x_f, y_f = point_2d
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        print(f"  ✓ Projected to 2D: ({x}, {y})")
        # Check if within image bounds
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), 8, (0, 255, 255), -1)      # Cyan filled circle
            cv2.circle(overlay, (x, y), 8, (255, 0, 0), 2)         # Blue outline
            cv2.putText(overlay, f"({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            print(f"Point is WITHIN image bounds")
            valid_projections += 1
        else:
            print(f"Projected point ({x}, {y}) is OUTSIDE image bounds ({w}x{h})")
        # Display the image
        cv2.imshow(f"Image {i} - Projected Tag Center", overlay)
        key = cv2.waitKey(500)  # 500ms per image
        if key == ord('q'):
            print("\nUser interrupted.")
            break
    cv2.destroyAllWindows()
    print(f"\n=== Dry Run Complete ===")
    print(f"Valid projections: {valid_projections}/{max_images}\n")

# from analyze_images_complete_eye_to_hand import t_base2tag
# The camera frame in the robot base frame from the calibration done in analyze_images_complete_eye_to_hand
# T_base2cam = np.array(
#  [[0.0200, 0.9997, -0.0131, 0.0977],
#  [0.2746, -0.0180, -0.9614, 0.2931],
#  [-0.9614, 0.0156, -0.2749, 1.2668],
#  [0.0000, 0.0000, 0.0000, 1.0000]])

# transform obtained from the 1/28 calibrated camera using 30/30 detected images 
T_base2cam = np.array(
[[-0.9986, -0.0080, -0.0516, 0.3074],
 [0.0518, -0.0291, -0.9982, 0.4037],
 [0.0065, -0.9995, 0.0295, 1.2454],
 [0.0000, 0.0000, 0.0000, 1.0000]]

 )
T_base2cam = np.load("/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/poses/cam2base_transform.npz")['arr_0']
# print("T_base2cam:\n", T_base2cam["arr_0"])
# T_base2cam = np.linalg.inv(T_base2cam)

T_base2cam = T_base2cam @ np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# print("T_base2cam:\n", T_base2cam)
# exit(0)

# camera_params = [716.3119506835938, 716.3119506835938, 655.386962890625, 397.7469787597656] # fx, fy, cx, cy
camera_params = [716.5634765625, 716.5634765625, 655.4454345703125 , 395.7761535644531]
fx = camera_params[0]
fy = camera_params[1]
cx = camera_params[2]
cy = camera_params[3]
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
max_images = 29

# Load saved transforms
t_base2gripper, r_base2gripper = load_saved_transforms("/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/poses/figure_eight_poses_1_28.npz")
t_base2tag, r_base2tag = get_center_tag_transforms(t_base2gripper, r_base2gripper)

# Plot the 3D positions of the tag locations
ground_truth_3d_plot(t_base2tag)

# Get and plot 2d projections
points_2d = get_image_points(max_images, t_base2tag, K, T_base2cam)
plot_2d_points(points_2d)

project_2d_points_on_images(max_images, points_2d)