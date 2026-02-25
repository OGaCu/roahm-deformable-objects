"""Visualize and align camera 3D point cloud with BOTH robot base frames (double arm).

This script:
- Loads a single RGB-D frame and EE poses from a double-arm capture sequence.
- Uses the calibrated T_base2cam for LEFT and RIGHT arms:
  - Shows both EE positions in the camera-frame point cloud.
  - Transforms the same camera point cloud into each robot base frame and shows it there.

Assumptions:
- Data layout from double_arm_capture.py:
    DATAPATH/captured_data_double_arm/{seq_name}/
        rgbd.npz           # color (N,H,W,3) BGR, depth (N,H,W) if Azure
        left_arm_poses.npz # arr_i = [x,y,z,qx,qy,qz,qw] in LEFT base frame
        right_arm_poses.npz# arr_i = [x,y,z,qx,qy,qz,qw] in RIGHT base frame
- Base-to-camera calibration saved by calculate_base_to_cam.py as:
    DATAPATH/poses/base2cam_transform_left.npz
    DATAPATH/poses/base2cam_transform_right.npz
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from apriltag_image import _camera_params_for


def _load_base2cam(datapath: Path, side: str) -> np.ndarray:
    """Load SE3 transform from base frame to camera frame for given side ('left' or 'right')."""
    fname = f"base2cam_transform_{side}.npz"
    path = datapath / "poses" / fname
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run calculate_base_to_cam.py with --side {side}.")
    T = np.load(path)["arr_0"]
    if T.shape != (4, 4):
        raise ValueError(f"{fname} has shape {T.shape}, expected (4,4)")
    return T


def _invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE3 transform."""
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ t
    return T_inv


def depth_to_pointcloud_cam(
    depth_mm: np.ndarray, K: np.ndarray, color_bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a color-aligned depth map (mm) to a colored point cloud in camera frame.

    depth_mm: (H, W) uint16 or float, depth in millimeters
    K: 3x3 intrinsics matrix
    color_bgr: (H, W, 3) uint8, aligned with depth (Azure color-aligned depth)
    Returns:
        pts_cam: (N, 3) float32 XYZ in camera frame
        colors: (N, 3) uint8 BGR colors for each point
    """
    depth = depth_mm.astype(np.float32) / 1000.0  # -> meters
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth.reshape(-1)
    mask = z > 0
    z = z[mask]
    u = u.reshape(-1)[mask]
    v = v.reshape(-1)[mask]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    colors_flat = color_bgr.reshape(-1, 3)[mask]
    return pts, colors_flat


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform T to Nx3 points (right-multiplied on homogeneous)."""
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    return (R @ pts.T).T + t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align camera 3D point cloud with BOTH left and right robot bases and visualize EEs."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects",
        help="Root data directory (where poses/ and captured_data_double_arm/ live).",
    )
    parser.add_argument(
        "--seq-name",
        type=str,
        required=True,
        help="Double-arm capture sequence name under captured_data_double_arm/.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to visualize (0-based, matches rgbd.npz and pose arrays).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of depth points to plot (subsample for speed).",
    )
    args = parser.parse_args()

    datapath = Path(args.datapath)
    seq_dir = datapath / "captured_data_double_arm" / args.seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    idx = args.index

    # Load base->cam for left and right, and their inverses (cam->base)
    T_left_base2cam = _load_base2cam(datapath, "left")
    T_right_base2cam = _load_base2cam(datapath, "right")
    T_cam2left = _invert_se3(T_left_base2cam)
    T_cam2right = _invert_se3(T_right_base2cam)

    # Intrinsics (Azure; must match the capture used)
    fx, fy, cx, cy = _camera_params_for("azure")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Load EE poses for this frame (left/right base frames)
    left_poses_npz = np.load(seq_dir / "left_arm_poses.npz")
    right_poses_npz = np.load(seq_dir / "right_arm_poses.npz")

    key = f"arr_{idx}"
    if key not in left_poses_npz:
        raise KeyError(f"{key} not found in left_arm_poses.npz")
    if key not in right_poses_npz:
        raise KeyError(f"{key} not found in right_arm_poses.npz")

    left_pose_vec = left_poses_npz[key]   # [x, y, z, qx, qy, qz, qw] in LEFT base
    right_pose_vec = right_poses_npz[key] # same in RIGHT base

    p_left_base = left_pose_vec[0:3]
    q_left = left_pose_vec[3:7]
    R_left_ee = Rotation.from_quat(q_left).as_matrix()

    p_right_base = right_pose_vec[0:3]
    q_right = right_pose_vec[3:7]
    R_right_ee = Rotation.from_quat(q_right).as_matrix()

    # Load RGB-D for this frame (color-aligned depth) from double-arm capture
    rgbd_path = seq_dir / "rgbd.npz"
    if not rgbd_path.exists():
        raise FileNotFoundError(f"rgbd.npz not found in {seq_dir}")
    rgbd = np.load(rgbd_path)
    colors = rgbd["color"]  # (N, H, W, 3) BGR uint8
    if idx < 0 or idx >= colors.shape[0]:
        raise IndexError(f"--index {idx} out of range; rgbd.npz has {colors.shape[0]} frames")
    color = colors[idx]

    depth_stack = rgbd.get("depth", None)
    if depth_stack is None:
        raise ValueError(f"No 'depth' array in {rgbd_path} (need color-aligned depth from Azure).")
    depth = depth_stack[idx]

    # Build camera-frame point cloud with color
    pts_cam, colors_cam_bgr = depth_to_pointcloud_cam(depth, K, color)

    # Foreground filter in camera frame:
    # keep only points with depth <= 1.5 m AND |X_cam| <= 0.5 m (others treated as background)
    # (X,Z are in meters in camera coordinates)
    foreground_mask = (pts_cam[:, 2] <= 1.5) & (np.abs(pts_cam[:, 0]) <= 0.5)
    pts_cam = pts_cam[foreground_mask]
    colors_cam_bgr = colors_cam_bgr[foreground_mask]

    if pts_cam.shape[0] == 0:
        raise RuntimeError("No foreground points after filtering; adjust thresholds.")

    # Split foreground into:
    # - pseudo-foreground: depth < 0.6 m
    # - real foreground: 0.6 m <= depth <= 1.5 m
    z_cam = pts_cam[:, 2]
    pseudo_mask = z_cam < 0.6
    real_mask = ~pseudo_mask
    pseudo_idx_all = np.nonzero(pseudo_mask)[0]
    real_idx_all = np.nonzero(real_mask)[0]

    max_pts = args.max_points
    if pts_cam.shape[0] <= max_pts:
        sel_idx = np.arange(pts_cam.shape[0])
    else:
        # 10% of max_points from pseudo foreground, 90% from real foreground (as available)
        target_pseudo = max(1, int(0.1 * max_pts))
        target_real = max_pts - target_pseudo

        n_pseudo = min(target_pseudo, len(pseudo_idx_all))
        n_real = min(target_real, len(real_idx_all))

        # If one group is short, top up from the other
        remaining = max_pts - (n_pseudo + n_real)
        if remaining > 0:
            if len(real_idx_all) - n_real > len(pseudo_idx_all) - n_pseudo:
                extra_real = min(remaining, len(real_idx_all) - n_real)
                n_real += extra_real
                remaining -= extra_real
        if remaining > 0 and len(pseudo_idx_all) - n_pseudo > 0:
            extra_pseudo = min(remaining, len(pseudo_idx_all) - n_pseudo)
            n_pseudo += extra_pseudo

        sel_idx_pseudo = (
            np.random.choice(pseudo_idx_all, size=n_pseudo, replace=False)
            if n_pseudo > 0 and len(pseudo_idx_all) > 0
            else np.array([], dtype=int)
        )
        sel_idx_real = (
            np.random.choice(real_idx_all, size=n_real, replace=False)
            if n_real > 0 and len(real_idx_all) > 0
            else np.array([], dtype=int)
        )
        sel_idx = np.concatenate([sel_idx_pseudo, sel_idx_real])

    pts_cam_plot = pts_cam[sel_idx]
    colors_cam_plot = colors_cam_bgr[sel_idx]

    # EE in camera frame (using each base->cam)
    p_left_base_h = np.hstack([p_left_base, 1.0])
    p_right_base_h = np.hstack([p_right_base, 1.0])
    p_left_cam = (T_left_base2cam @ p_left_base_h)[:3]
    p_right_cam = (T_right_base2cam @ p_right_base_h)[:3]

    # Transform entire (foreground) cloud into each base frame
    pts_left_base = transform_points(T_cam2left, pts_cam)
    pts_right_base = transform_points(T_cam2right, pts_cam)
    pts_left_base_plot = pts_left_base[sel_idx]
    pts_right_base_plot = pts_right_base[sel_idx]
    colors_left_base_plot = colors_cam_bgr[sel_idx]
    colors_right_base_plot = colors_cam_bgr[sel_idx]

    # Convert to millimeters for visualization
    scale_mm = 1000.0
    pts_cam_plot_mm = pts_cam_plot * scale_mm
    pts_left_base_plot_mm = pts_left_base_plot * scale_mm
    pts_right_base_plot_mm = pts_right_base_plot * scale_mm
    p_left_cam_mm = p_left_cam * scale_mm
    p_right_cam_mm = p_right_cam * scale_mm
    p_left_base_mm = p_left_base * scale_mm
    p_right_base_mm = p_right_base * scale_mm

    # Visualization 1: EE positions in camera-frame point cloud (units: mm)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.set_facecolor("black")
    colors_cam_rgb = colors_cam_plot[:, ::-1] / 255.0  # BGR -> RGB
    ax1.scatter(
        pts_cam_plot_mm[:, 0],
        pts_cam_plot_mm[:, 1],
        pts_cam_plot_mm[:, 2],
        c=colors_cam_rgb,
        s=0.5,
        alpha=0.3,
    )
    # Origin and both EEs in camera frame
    ax1.scatter(0.0, 0.0, 0.0, c="blue", s=40, label="origin (cam)")
    ax1.scatter(
        p_left_cam_mm[0],
        p_left_cam_mm[1],
        p_left_cam_mm[2],
        c="red",
        s=60,
        label="left EE (cam frame)",
    )
    ax1.scatter(
        p_right_cam_mm[0],
        p_right_cam_mm[1],
        p_right_cam_mm[2],
        c="cyan",
        s=60,
        label="right EE (cam frame)",
    )
    # Origin axis orientation (camera frame)
    axis_len_cam_mm = 100.0
    ax1.plot([0, axis_len_cam_mm], [0, 0], [0, 0], color="r")  # X_cam
    ax1.plot([0, 0], [0, axis_len_cam_mm], [0, 0], color="g")  # Y_cam
    ax1.plot([0, 0], [0, 0], [0, axis_len_cam_mm], color="b")  # Z_cam

    ax1.set_xlabel("X_cam (mm)", color="white")
    ax1.set_ylabel("Y_cam (mm)", color="white")
    ax1.set_zlabel("Z_cam (mm)", color="white")
    ax1.set_title(f"Camera-frame point cloud with left/right EEs (frame {idx})", color="white")
    ax1.tick_params(colors="white")
    ax1.legend()

    # Visualization 2: point cloud transformed into LEFT base frame, with LEFT EE (units: mm)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_facecolor("black")
    colors_left_base_rgb = colors_left_base_plot[:, ::-1] / 255.0
    ax2.scatter(
        pts_left_base_plot_mm[:, 0],
        pts_left_base_plot_mm[:, 1],
        pts_left_base_plot_mm[:, 2],
        c=colors_left_base_rgb,
        s=0.5,
        alpha=0.3,
    )
    ax2.scatter(0.0, 0.0, 0.0, c="blue", s=40, label="origin (left base)")
    ax2.scatter(
        p_left_base_mm[0],
        p_left_base_mm[1],
        p_left_base_mm[2],
        c="red",
        s=60,
        label="left EE (left base)",
    )
    # LEFT EE orientation axes in left base frame
    axis_len_left_mm = 50.0
    left_ee_axes_mm = R_left_ee @ (axis_len_left_mm * np.eye(3))
    colors_axes = ["r", "g", "b"]
    for i in range(3):
        ax2.plot(
            [p_left_base_mm[0], p_left_base_mm[0] + left_ee_axes_mm[0, i]],
            [p_left_base_mm[1], p_left_base_mm[1] + left_ee_axes_mm[1, i]],
            [p_left_base_mm[2], p_left_base_mm[2] + left_ee_axes_mm[2, i]],
            color=colors_axes[i],
        )
    # Origin axis orientation in left base frame
    axis_len_left_origin_mm = 100.0
    ax2.plot([0, axis_len_left_origin_mm], [0, 0], [0, 0], color="r")
    ax2.plot([0, 0], [0, axis_len_left_origin_mm], [0, 0], color="g")
    ax2.plot([0, 0], [0, 0], [0, axis_len_left_origin_mm], color="b")

    ax2.set_xlabel("X_left_base (mm)", color="white")
    ax2.set_ylabel("Y_left_base (mm)", color="white")
    ax2.set_zlabel("Z_left_base (mm)", color="white")
    ax2.set_title(f"Left-base point cloud with left EE pose (frame {idx})", color="white")
    ax2.tick_params(colors="white")
    ax2.legend()

    # Visualization 3: point cloud transformed into RIGHT base frame, with RIGHT EE (units: mm)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_facecolor("black")
    colors_right_base_rgb = colors_right_base_plot[:, ::-1] / 255.0
    ax3.scatter(
        pts_right_base_plot_mm[:, 0],
        pts_right_base_plot_mm[:, 1],
        pts_right_base_plot_mm[:, 2],
        c=colors_right_base_rgb,
        s=0.5,
        alpha=0.3,
    )
    ax3.scatter(0.0, 0.0, 0.0, c="blue", s=40, label="origin (right base)")
    ax3.scatter(
        p_right_base_mm[0],
        p_right_base_mm[1],
        p_right_base_mm[2],
        c="cyan",
        s=60,
        label="right EE (right base)",
    )
    # RIGHT EE orientation axes in right base frame
    axis_len_right_mm = 50.0
    right_ee_axes_mm = R_right_ee @ (axis_len_right_mm * np.eye(3))
    for i in range(3):
        ax3.plot(
            [p_right_base_mm[0], p_right_base_mm[0] + right_ee_axes_mm[0, i]],
            [p_right_base_mm[1], p_right_base_mm[1] + right_ee_axes_mm[1, i]],
            [p_right_base_mm[2], p_right_base_mm[2] + right_ee_axes_mm[2, i]],
            color=colors_axes[i],
        )
    # Origin axis orientation in right base frame
    axis_len_right_origin_mm = 100.0
    ax3.plot([0, axis_len_right_origin_mm], [0, 0], [0, 0], color="r")
    ax3.plot([0, 0], [0, axis_len_right_origin_mm], [0, 0], color="g")
    ax3.plot([0, 0], [0, 0], [0, axis_len_right_origin_mm], color="b")

    ax3.set_xlabel("X_right_base (mm)", color="white")
    ax3.set_ylabel("Y_right_base (mm)", color="white")
    ax3.set_zlabel("Z_right_base (mm)", color="white")
    ax3.set_title(f"Right-base point cloud with right EE pose (frame {idx})", color="white")
    ax3.tick_params(colors="white")
    ax3.legend()

    plt.show()


if __name__ == "__main__":
    main()

