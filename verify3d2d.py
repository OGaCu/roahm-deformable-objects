import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from calculate_base_to_cam import gripper2tag
from azure_intrinsics import azure_intrinsics
from apriltag_image import _camera_params_for

# Tag positions and rotations
def load_saved_transforms(transform_file: str, num_transforms: int | None = None):
    """Load base-to-gripper poses from npz (arr_0, arr_1, ...). If num_transforms is None, load all."""
    np_info = np.load(transform_file)
    arr_keys = [k for k in np_info.files if k.startswith("arr_")]
    arr_keys.sort(key=lambda k: int(k.split("_")[1]))
    if num_transforms is not None:
        arr_keys = arr_keys[:num_transforms]
    translations = []
    rotations = []
    for k in arr_keys:
        np_stuff = np_info[k]
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

        t_base2tag_mat = t_base2gripper_mat# @ gripper2tag
        r_base2tag.append(t_base2tag_mat[0:3, 0:3])
        t_base2tag.append(t_base2tag_mat[0:3, 3])
    return t_base2tag, r_base2tag


def plot_3d_trajectory(t_base2tag):
    """Show 3D trajectory in an interactive window: green tail, orange current (last) position."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xs = np.array([t[0] for t in t_base2tag])
    ys = np.array([t[1] for t in t_base2tag])
    zs = np.array([t[2] for t in t_base2tag])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Trajectory line and tail points in green
    ax.plot(xs, ys, zs, color="green", alpha=0.7, linewidth=1)
    ax.scatter(xs, ys, zs, c="green", s=8, alpha=0.8)
    # Current (last) position in orange
    if len(xs) > 0:
        ax.scatter(xs[-1], ys[-1], zs[-1], c="orange", s=120, edgecolors="none", zorder=5)
    all_x, all_y, all_z = xs, ys, zs
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) or 0.1
    mid_x = (x_min + x_max) / 2.0
    mid_y = (y_min + y_max) / 2.0
    mid_z = (z_min + z_max) / 2.0
    ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
    ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
    ax.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D trajectory (green=path, orange=current)")
    plt.tight_layout()
    plt.show()

def project_3d_to_2d(point_3d, K, T_base2cam, use_cv2=True):
    """
    Project a 3D point from base frame to 2D image coordinates.
    T_base2cam is a transformation matrix of the camera frame in the base frame coordinates.
    Returns (x, y) or None if behind camera.
    """
    if not use_cv2:
        point_base_homog = np.hstack([point_3d, 1.0])
        point_cam = (T_base2cam @ point_base_homog)[:3]
        # Check if point is in front of camera
        if point_cam[2] <= 0:
            print(f"Warning: Point behind camera (z={point_cam[2]})")
            return None
        # Project using intrinsics
        point_img_homog = K @ point_cam
        point_2d = point_img_homog[:2] / point_img_homog[2]
    else:
        point_base_homog = np.hstack([np.asarray(point_3d).reshape(3), 1.0])
        point_cam = (T_base2cam @ point_base_homog)[:3]
        if point_cam[2] <= 0:
            return None
        point_3d = point_3d.reshape(1, 1, 3)
        r_vec, _ = cv2.Rodrigues(T_base2cam[:3, :3])
        t_vec = T_base2cam[:3, 3]
        point_2d, _ = cv2.projectPoints(point_3d, r_vec, t_vec, K, distCoeffs=None)
        return point_2d.reshape(2)


def get_all_2d_points(t_base2tag, K, T_base2cam):
    """One 2D point per pose; None if behind camera. Same length as t_base2tag."""
    points_2d = []
    for i in range(len(t_base2tag)):
        result = project_3d_to_2d(t_base2tag[i], K, T_base2cam, use_cv2=True)
        points_2d.append(result)
    return points_2d


def write_trajectory_video(
    datapath: str,
    points_2d: list,
    tail_length: int,
    downsample: int,
    output_path: str,
    fps: float = 10.0,
):
    """
    Write an .mp4 where each frame is the camera image with the trajectory tail overlaid.
    Tail = green small points; current position = orange. Frame i uses image at original index i * downsample.
    """
    n_frames = len(points_2d)
    if n_frames == 0:
        print("No frames to write.")
        return
    first_img_path = f"{datapath}/images/single_arm_image_pose_0.png"
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Cannot load {first_img_path}; aborting video.")
        return
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return
    green = (0, 255, 0)   # BGR tail
    orange = (0, 165, 255)  # BGR current position
    radius_tail = 2
    radius_current = 4
    thickness = -1

    for i in range(n_frames):
        orig_idx = i * downsample
        img_path = f"{datapath}/images/single_arm_image_pose_{orig_idx}.png"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        overlay = img.copy()
        start_idx = max(0, i - tail_length + 1)
        # Tail: green points (all in tail including current, then overwrite current with orange)
        for j in range(start_idx, i + 1):
            pt = points_2d[j]
            if pt is None:
                continue
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
            if 0 <= x < w and 0 <= y < h:
                if j == i:
                    cv2.circle(overlay, (x, y), radius_current, orange, thickness)
                else:
                    cv2.circle(overlay, (x, y), radius_tail, green, thickness)
        writer.write(overlay)
    writer.release()
    print(f"Saved trajectory video: {output_path} ({n_frames} frames)")

parser = argparse.ArgumentParser(description="Verify hand-eye calibration: project EE trajectory to 2D and save as video.")
parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame (default: 1 = no downsample)")
parser.add_argument("--tail-length", type=int, default=60, help="Trajectory tail length in frames (default: 60)")
parser.add_argument("--output", type=str, default="traj_2d_verify.mp4", help="Output video path (default: traj_2d_verify.mp4)")
parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
args = parser.parse_args()

DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"

T_base2cam = np.load(f"{DATAPATH}/poses/base2cam_transform.npz")["arr_0"]
print("T_base2cam:\n", T_base2cam)

camera_params = _camera_params_for("azure")
fx, fy = camera_params[0], camera_params[1]
cx, cy = camera_params[2], camera_params[3]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

right_to_left_transform = np.eye(4)

# Load all poses (no max)
t_base2gripper, r_base2gripper = load_saved_transforms(f"{DATAPATH}/poses/single_arm_poses.npz")
t_base2tag, r_base2tag = get_center_tag_transforms(t_base2gripper, r_base2gripper)

# Transform base2tag from right arm frame to left arm frame
t_base2tag_left = []
r_base2tag_left = []
for r, t in zip(r_base2tag, t_base2tag):
    base2tag_mat = np.eye(4)
    base2tag_mat[0:3, 0:3] = r
    base2tag_mat[0:3, 3] = t
    base2tag_left_mat = right_to_left_transform @ base2tag_mat
    r_base2tag_left.append(base2tag_left_mat[0:3, 0:3])
    t_base2tag_left.append(base2tag_left_mat[0:3, 3])
t_base2tag = t_base2tag_left

# Downsample: use every Nth frame
downsample = max(1, args.downsample)
t_base2tag_ds = t_base2tag[::downsample]
print(f"Using {len(t_base2tag_ds)} frames (downsample={downsample}, total poses={len(t_base2tag)})")

# 2D projections for downsampled frames
points_2d = get_all_2d_points(t_base2tag_ds, K, T_base2cam)
valid = sum(1 for p in points_2d if p is not None)
print(f"Valid 2D projections: {valid}/{len(points_2d)}")

# Video: camera image as background, green tail overlay (small points, no border)
out_path = args.output if args.output.endswith(".mp4") else f"{args.output}.mp4"
if not out_path.startswith("/") and "/" not in out_path:
    out_path = f"{DATAPATH}/{out_path}"
write_trajectory_video(
    DATAPATH,
    points_2d,
    tail_length=args.tail_length,
    downsample=downsample,
    output_path=out_path,
    fps=args.fps,
)

# Show 3D trajectory in interactive window (green=path, orange=current)
print("Opening 3D trajectory window (close to exit)...")
plot_3d_trajectory(t_base2tag_ds)