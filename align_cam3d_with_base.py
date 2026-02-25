"""Visualize and align camera 3D point cloud with robot base frame.

This script:
- Loads a single RGB-D frame and EE pose from the single-arm capture outputs.
- Uses the calibrated T_base2cam to:
  - Show the EE position in the camera-frame point cloud.
  - Transform the camera point cloud into the robot base frame and show it there.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from apriltag_image import _camera_params_for


def load_base2cam(datapath: Path) -> np.ndarray:
    """Load SE3 transform from base frame to camera frame."""
    T = np.load(datapath / "poses" / "base2cam_transform_right.npz")["arr_0"]
    if T.shape != (4, 4):
        raise ValueError(f"base2cam_transform has shape {T.shape}, expected (4,4)")
    return T


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


def project_base_point_to_image(
    p_base: np.ndarray, T_base2cam: np.ndarray, K: np.ndarray
) -> tuple[float, float] | None:
    """Project a base-frame point into image coordinates using T_base2cam and K."""
    p_base_h = np.hstack([p_base, 1.0])
    p_cam = (T_base2cam @ p_base_h)[:3]
    if p_cam[2] <= 0:
        return None
    uvs = K @ p_cam
    return float(uvs[0] / uvs[2]), float(uvs[1] / uvs[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align camera 3D point cloud with robot base frame and visualize EE."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects",
        help="Root data directory (where poses/ and images/ live).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to visualize (matches single_arm_image_pose_{index}.png and rgbd npz).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of depth points to plot (subsample for speed).",
    )
    args = parser.parse_args()

    datapath = Path(args.datapath)
    idx = args.index

    # Load base->cam and its inverse
    T_base2cam = load_base2cam(datapath)
    R_bc = T_base2cam[0:3, 0:3]
    t_bc = T_base2cam[0:3, 3]
    R_cb = R_bc.T
    t_cb = -R_cb @ t_bc
    T_cam2base = np.eye(4)
    T_cam2base[0:3, 0:3] = R_cb
    T_cam2base[0:3, 3] = t_cb

    # Intrinsics (must match the capture used)
    fx, fy, cx, cy = _camera_params_for("azure")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Load EE pose for this frame (base frame)
    poses_npz = np.load(datapath / "poses" / "single_arm_poses.npz")
    key = f"arr_{idx}"
    if key not in poses_npz:
        raise KeyError(f"{key} not found in single_arm_poses.npz")
    pose_vec = poses_npz[key]  # [x, y, z, qx, qy, qz, qw]
    p_base = pose_vec[0:3]
    q = pose_vec[3:7]
    R_base_ee = Rotation.from_quat(q).as_matrix()

    # Load RGB-D for this frame (color-aligned depth)
    rgbd_path = datapath / "images" / f"single_arm_rgbd_pose_{idx}.npz"
    rgbd = np.load(rgbd_path)
    color = rgbd["color"]  # (H, W, 3) BGR uint8
    depth = rgbd.get("depth", None)
    if depth is None:
        raise ValueError(f"No 'depth' in {rgbd_path} (need color-aligned depth).")

    # Build camera-frame point cloud with color
    pts_cam, colors_cam_bgr = depth_to_pointcloud_cam(depth, K, color)

    # Foreground filter in camera frame:
    # keep only points with depth <= 1.5 m AND |X_cam| <= 0.5 m (others treated as background)
    # (X,Z are in meters in camera coordinates)
    foreground_mask = (pts_cam[:, 2] <= 1.5) & (np.abs(pts_cam[:, 0]) <= 0.5)
    pts_cam = pts_cam[foreground_mask]
    colors_cam_bgr = colors_cam_bgr[foreground_mask]

    # Split foreground into:
    # - pseudo-foreground: depth < 0.6 m
    # - real foreground: 0.6 m <= depth <= 1.5 m
    if pts_cam.shape[0] == 0:
        raise RuntimeError("No foreground points after filtering; adjust thresholds.")

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

    # EE in camera frame (using base->cam)
    p_base_h = np.hstack([p_base, 1.0])
    p_cam = (T_base2cam @ p_base_h)[:3]

    # Transform entire (foreground) cloud into base frame
    pts_base = transform_points(T_cam2base, pts_cam)
    pts_base_plot = pts_base[sel_idx]
    colors_base_plot = colors_cam_bgr[sel_idx]

    # Convert to millimeters for visualization
    scale_mm = 1000.0
    pts_cam_plot_mm = pts_cam_plot * scale_mm
    pts_base_plot_mm = pts_base_plot * scale_mm
    p_cam_mm = p_cam * scale_mm
    p_base_mm = p_base * scale_mm

    # Visualization 1: EE in camera-frame point cloud (units: mm)
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
    # EE in cam frame (red) and origin (blue), with origin axes
    ax1.scatter(0.0, 0.0, 0.0, c="blue", s=40, label="origin (cam)")
    ax1.scatter(p_cam_mm[0], p_cam_mm[1], p_cam_mm[2], c="red", s=60, label="EE (cam frame)")
    # Origin axis orientation (camera frame)
    axis_len_cam_mm = 100.0
    ax1.plot([0, axis_len_cam_mm], [0, 0], [0, 0], color="r")  # X_cam
    ax1.plot([0, 0], [0, axis_len_cam_mm], [0, 0], color="g")  # Y_cam
    ax1.plot([0, 0], [0, 0], [0, axis_len_cam_mm], color="b")  # Z_cam

    ax1.set_xlabel("X_cam (mm)", color="white")
    ax1.set_ylabel("Y_cam (mm)", color="white")
    ax1.set_zlabel("Z_cam (mm)", color="white")
    ax1.set_title(f"Camera-frame point cloud with EE (frame {idx})", color="white")
    ax1.tick_params(colors="white")
    ax1.legend()

    # Visualization 2: point cloud transformed into base frame, with EE in base (units: mm)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_facecolor("black")
    colors_base_rgb = colors_base_plot[:, ::-1] / 255.0
    ax2.scatter(
        pts_base_plot_mm[:, 0],
        pts_base_plot_mm[:, 1],
        pts_base_plot_mm[:, 2],
        c=colors_base_rgb,
        s=0.5,
        alpha=0.3,
    )
    # EE in base frame (red) and origin (blue)
    ax2.scatter(0.0, 0.0, 0.0, c="blue", s=40, label="origin (base)")
    ax2.scatter(p_base_mm[0], p_base_mm[1], p_base_mm[2], c="red", s=60, label="EE (base frame)")
    # EE orientation axes in base frame, centered at EE, length in mm
    axis_len_base_mm = 50.0
    ee_axes_mm = R_base_ee @ (axis_len_base_mm * np.eye(3))
    colors = ["r", "g", "b"]
    for i in range(3):
        ax2.plot(
            [p_base_mm[0], p_base_mm[0] + ee_axes_mm[0, i]],
            [p_base_mm[1], p_base_mm[1] + ee_axes_mm[1, i]],
            [p_base_mm[2], p_base_mm[2] + ee_axes_mm[2, i]],
            color=colors[i],
        )
    # Origin axis orientation in base frame
    axis_len_base_origin_mm = 100.0
    ax2.plot([0, axis_len_base_origin_mm], [0, 0], [0, 0], color="r")
    ax2.plot([0, 0], [0, axis_len_base_origin_mm], [0, 0], color="g")
    ax2.plot([0, 0], [0, 0], [0, axis_len_base_origin_mm], color="b")

    ax2.set_xlabel("X_base (mm)", color="white")
    ax2.set_ylabel("Y_base (mm)", color="white")
    ax2.set_zlabel("Z_base (mm)", color="white")
    ax2.set_title(f"Base-frame point cloud with EE pose (frame {idx})", color="white")
    ax2.tick_params(colors="white")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()

