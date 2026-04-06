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
    frame_delay: int,
    output_path: str,
    fps: float = 10.0,
    points_2d_second: list | None = None,
    image_name_pattern: str = "single_arm_image_pose_{i}.png",
):
    """
    Write an .mp4 with trajectory tail(s) overlaid. If points_2d_second is given, draw both
    (first = green/orange, second = blue/cyan).
    Frame i uses image at index i * downsample + frame_delay.
    Positive frame_delay shifts EE overlays later in the image sequence.
    """
    n_frames = len(points_2d)
    if points_2d_second is not None and len(points_2d_second) != n_frames:
        n_frames = min(n_frames, len(points_2d_second))
    if n_frames == 0:
        print("No frames to write.")
        return
    first_img_path = f"{datapath}/images/{image_name_pattern.format(i=max(0, frame_delay))}"
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
    green = (0, 255, 0)
    orange = (0, 165, 255)
    blue_tail = (255, 128, 0)   # BGR second trail
    cyan_current = (255, 255, 0)  # BGR second current
    radius_tail = 2
    radius_current = 4
    thickness = -1

    for i in range(n_frames):
        orig_idx = i * downsample + frame_delay
        img_path = f"{datapath}/images/{image_name_pattern.format(i=orig_idx)}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        overlay = img.copy()
        start_idx = max(0, i - tail_length + 1)

        def draw_trail(pts, color_tail, color_current):
            for j in range(start_idx, min(i + 1, len(pts))):
                pt = pts[j]
                if pt is None:
                    continue
                x = int(round(float(pt[0])))
                y = int(round(float(pt[1])))
                if 0 <= x < w and 0 <= y < h:
                    if j == i:
                        cv2.circle(overlay, (x, y), radius_current, color_current, thickness)
                    else:
                        cv2.circle(overlay, (x, y), radius_tail, color_tail, thickness)

        draw_trail(points_2d, green, orange)
        if points_2d_second is not None:
            draw_trail(points_2d_second, blue_tail, cyan_current)
        writer.write(overlay)
    writer.release()
    print(f"Saved trajectory video: {output_path} ({n_frames} frames)")

parser = argparse.ArgumentParser(description="Verify hand-eye calibration: project EE trajectory to 2D and save as video.")
parser.add_argument("--mode", type=str, choices=["single", "dual"], default="single",
    help="single: one arm (single_arm_poses.npz). dual: both arms (left_arm_poses + right_arm_poses), project both with right_to_left transform.")
parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame (default: 1 = no downsample)")
parser.add_argument("--frame-delay", type=int, default=0, help="Image index offset for overlay. First EE point is drawn on image index=frame-delay.")
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

# Transform from right-arm base frame to left-arm base frame (used to express right-arm tag in left base for projection).
# Uncomment to use: rotation by pi around Z + translation along X.
initial_rotation = np.pi
turn_rotation = np.array([
    [np.cos(initial_rotation), -np.sin(initial_rotation), 0.0],
    [np.sin(initial_rotation), np.cos(initial_rotation), 0.0],
    [0.0, 0.0, 1.0],
])
right_to_left_transform = np.eye(4)
right_to_left_transform[0:3, 0:3] = turn_rotation
right_to_left_transform[0:3, 3] = np.array([1.258, 0.0, 0.0])
print("right_to_left_transform:\n", right_to_left_transform)

downsample = max(1, args.downsample)
out_path = args.output if args.output.endswith(".mp4") else f"{args.output}.mp4"
if not out_path.startswith("/") and "/" not in out_path:
    out_path = f"{DATAPATH}/{out_path}"

if args.mode == "single":
    # Single arm: load single_arm_poses, apply right_to_left so tag is in left base, project.
    t_base2gripper, r_base2gripper = load_saved_transforms(f"{DATAPATH}/poses/single_arm_poses.npz")
    t_base2tag, r_base2tag = get_center_tag_transforms(t_base2gripper, r_base2gripper)
    t_base2tag_left = []
    for r, t in zip(r_base2tag, t_base2tag):
        base2tag_mat = np.eye(4)
        base2tag_mat[0:3, 0:3] = r
        base2tag_mat[0:3, 3] = t
        base2tag_left_mat = base2tag_mat
        t_base2tag_left.append(base2tag_left_mat[0:3, 3])
    t_base2tag_ds = t_base2tag_left[::downsample]
    points_2d = get_all_2d_points(t_base2tag_ds, K, T_base2cam)
    points_2d_second = None
    image_pattern = "single_arm_image_pose_{i}.png"
    t_base2tag_ds_left = t_base2tag_ds
    t_base2tag_ds_right = None
else:
    # Dual arm: load left_arm_poses and right_arm_poses. Use right tag in right base as-is; transform left tag to right base (left_to_right) for projection (T_base2cam is in right base).
    left_to_right_transform = np.linalg.inv(right_to_left_transform)
    t_base2left_grip, r_base2left_grip = load_saved_transforms(f"{DATAPATH}/poses/left_arm_poses.npz")
    t_base2right_grip, r_base2right_grip = load_saved_transforms(f"{DATAPATH}/poses/right_arm_poses.npz")
    t_base2tag_left_list, r_base2tag_left = get_center_tag_transforms(t_base2left_grip, r_base2left_grip)
    t_base2tag_right_list, _ = get_center_tag_transforms(t_base2right_grip, r_base2right_grip)
    # Left tag in left base -> transform to right base for projection.
    t_base2tag_left_in_right = []
    for r, t in zip(r_base2tag_left, t_base2tag_left_list):
        base2tag_mat = np.eye(4)
        base2tag_mat[0:3, 0:3] = r
        base2tag_mat[0:3, 3] = t
        in_right = left_to_right_transform @ base2tag_mat
        t_base2tag_left_in_right.append(in_right[0:3, 3])
    n_dual = min(len(t_base2tag_left_in_right), len(t_base2tag_right_list))
    t_base2tag_left_in_right = t_base2tag_left_in_right[:n_dual]
    t_base2tag_right_list = t_base2tag_right_list[:n_dual]
    t_base2tag_ds_left = t_base2tag_left_in_right[::downsample]
    t_base2tag_ds_right = t_base2tag_right_list[::downsample]
    # First trail = left arm (in right base), second trail = right arm (in right base).
    points_2d = get_all_2d_points(t_base2tag_ds_left, K, T_base2cam)
    points_2d_second = get_all_2d_points(t_base2tag_ds_right, K, T_base2cam)
    image_pattern = "double_arm_image_{i}.png"

n_frames = len(points_2d)
valid = sum(1 for p in points_2d if p is not None)
print(f"Mode: {args.mode}. Using {n_frames} frames (downsample={downsample}). Valid left projections: {valid}/{n_frames}")
if points_2d_second is not None:
    valid_r = sum(1 for p in points_2d_second if p is not None)
    print(f"Valid right projections: {valid_r}/{len(points_2d_second)}")

write_trajectory_video(
    DATAPATH,
    points_2d,
    tail_length=args.tail_length,
    downsample=downsample,
    frame_delay=args.frame_delay,
    output_path=out_path,
    fps=args.fps,
    points_2d_second=points_2d_second,
    image_name_pattern=image_pattern,
)

# 3D trajectory window
print("Opening 3D trajectory window (close to exit)...")
if args.mode == "single":
    plot_3d_trajectory(t_base2tag_ds)
else:
    # Plot both arms: left green/orange, right blue/cyan
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xs_l = np.array([t[0] for t in t_base2tag_ds_left])
    ys_l = np.array([t[1] for t in t_base2tag_ds_left])
    zs_l = np.array([t[2] for t in t_base2tag_ds_left])
    xs_r = np.array([t[0] for t in t_base2tag_ds_right])
    ys_r = np.array([t[1] for t in t_base2tag_ds_right])
    zs_r = np.array([t[2] for t in t_base2tag_ds_right])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs_l, ys_l, zs_l, color="green", alpha=0.7, linewidth=1, label="left (in right base)")
    ax.scatter(xs_l, ys_l, zs_l, c="green", s=8, alpha=0.8)
    if len(xs_l) > 0:
        ax.scatter(xs_l[-1], ys_l[-1], zs_l[-1], c="orange", s=120, edgecolors="none", zorder=5)
    ax.plot(xs_r, ys_r, zs_r, color="blue", alpha=0.7, linewidth=1, label="right")
    ax.scatter(xs_r, ys_r, zs_r, c="blue", s=8, alpha=0.8)
    if len(xs_r) > 0:
        ax.scatter(xs_r[-1], ys_r[-1], zs_r[-1], c="cyan", s=120, edgecolors="none", zorder=5)
    all_x = np.hstack([xs_l, xs_r])
    all_y = np.hstack([ys_l, ys_r])
    all_z = np.hstack([zs_l, zs_r])
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
    ax.set_title("3D trajectories (green=left in right base, blue=right)")
    ax.legend()
    plt.tight_layout()
    plt.show()