#!/usr/bin/env python3
"""Compute nearest point-cloud distance stats to EE over a full single-arm sequence.

Reads new-format capture outputs:
  captured_data_single_arm/<seq_name>/
    rgbd.npz            # color (N,H,W,3), depth (N,H,W) uint16 mm (Azure)
    single_arm_poses.npz# arr_i = [x,y,z,qx,qy,qz,qw] in base frame

For each frame:
  1) Lift depth to camera-frame cloud.
  2) Transform cloud to base frame.
  3) Find nearest cloud point to EE position.
  4) Store component-wise deltas: nn - ee  (dx, dy, dz) in meters.

Then prints statistics over the sequence:
  - mean signed dx/dy/dz
  - mean absolute |dx|/|dy|/|dz|
  - std, RMSE, median, p95 for each axis
  - Euclidean nearest distance stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apriltag_image import _camera_params_for


def _load_base2cam(transform_npz: Path) -> np.ndarray:
    data = np.load(transform_npz)
    T = data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
    if T.shape != (4, 4):
        raise ValueError(f"base2cam transform must be (4,4), got {T.shape} from {transform_npz}")
    return T


def _load_pose_vecs(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    arr_keys = [k for k in data.files if k.startswith("arr_")]
    arr_keys.sort(key=lambda k: int(k.split("_", 1)[1]))
    if not arr_keys:
        raise ValueError(f"No arr_* keys found in {npz_path}")
    vecs = np.stack([np.asarray(data[k]) for k in arr_keys], axis=0)
    if vecs.ndim != 2 or vecs.shape[1] < 3:
        raise ValueError(f"Unexpected pose array shape {vecs.shape} from {npz_path}")
    return vecs


def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _depth_to_pointcloud_cam(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    depth = depth_mm.astype(np.float32) / 1000.0  # m
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
    return np.stack([x, y, z], axis=1)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _axis_stats(vals: np.ndarray) -> dict:
    abs_vals = np.abs(vals)
    return {
        "mean": float(np.mean(vals)),
        "mean_abs": float(np.mean(abs_vals)),
        "std": float(np.std(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "median": float(np.median(vals)),
        "p95_abs": float(np.percentile(abs_vals, 95.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-wide EE-to-nearest-cloud-point statistics.")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects",
        help="Root data directory.",
    )
    parser.add_argument("--seq-name", type=str, required=True, help="Sequence under captured_data_single_arm/.")
    parser.add_argument("--camera", type=str, choices=["azure", "zed"], default="azure", help="Intrinsics to use.")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="right", help="base2cam side.")
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Directory containing base2cam_transform_{left,right}.npz; default datapath/poses.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive). -1 uses full sequence.")
    parser.add_argument("--max-depth-m", type=float, default=1.5, help="Depth upper bound for cloud points.")
    parser.add_argument("--max-x-cam-m", type=float, default=0.5, help="Camera-frame |X| bound.")
    parser.add_argument(
        "--around-ee-radius-m",
        type=float,
        default=-1.0,
        help="If >0, keep only points within this radius of EE (camera frame).",
    )
    parser.add_argument("--save-csv", type=str, default=None, help="Optional CSV path to save per-frame deltas.")
    args = parser.parse_args()

    datapath = Path(args.datapath)
    seq_dir = datapath / "captured_data_single_arm" / args.seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    rgbd_path = seq_dir / "rgbd.npz"
    poses_path = seq_dir / "single_arm_poses.npz"
    if not rgbd_path.exists():
        raise FileNotFoundError(f"Missing {rgbd_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing {poses_path}")

    calib_dir = Path(args.calib_dir) if args.calib_dir else (datapath / "poses")
    base2cam_path = calib_dir / f"base2cam_transform_{args.side}.npz"
    if not base2cam_path.exists():
        fallback = calib_dir / "base2cam_transform.npz"
        if fallback.exists():
            base2cam_path = fallback
        else:
            raise FileNotFoundError(f"Missing {base2cam_path} (and fallback {fallback})")

    T_base2cam = _load_base2cam(base2cam_path)
    T_cam2base = _invert_se3(T_base2cam)

    fx, fy, cx, cy = _camera_params_for(args.camera)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgbd = np.load(rgbd_path)
    colors = rgbd["color"]
    depth_stack = rgbd.get("depth", None)
    if depth_stack is None:
        raise ValueError(f"No depth in {rgbd_path}")
    poses = _load_pose_vecs(poses_path)

    n_total = min(depth_stack.shape[0], poses.shape[0], colors.shape[0])
    start = max(0, args.start)
    end = n_total if args.end < 0 else min(args.end, n_total)
    if start >= end:
        raise ValueError(f"Invalid range start={start}, end={end}, n_total={n_total}")

    frame_ids: list[int] = []
    deltas_xyz: list[np.ndarray] = []
    dists: list[float] = []

    skipped_empty = 0
    for i in range(start, end):
        depth = depth_stack[i]
        p_base = poses[i, 0:3]
        p_cam = (T_base2cam @ np.hstack([p_base, 1.0]))[:3]

        pts_cam = _depth_to_pointcloud_cam(depth, K)
        if pts_cam.shape[0] == 0:
            skipped_empty += 1
            continue

        mask = (pts_cam[:, 2] <= args.max_depth_m) & (np.abs(pts_cam[:, 0]) <= args.max_x_cam_m)
        if args.around_ee_radius_m > 0:
            mask &= np.linalg.norm(pts_cam - p_cam.reshape(1, 3), axis=1) <= args.around_ee_radius_m
        pts_cam = pts_cam[mask]
        if pts_cam.shape[0] == 0:
            skipped_empty += 1
            continue

        pts_base = _transform_points(T_cam2base, pts_cam)
        deltas = pts_base - p_base.reshape(1, 3)  # nn - ee
        norms = np.linalg.norm(deltas, axis=1)
        nn_idx = int(np.argmin(norms))

        frame_ids.append(i)
        deltas_xyz.append(deltas[nn_idx])
        dists.append(float(norms[nn_idx]))

    if not deltas_xyz:
        raise RuntimeError("No valid frames left after filtering; nothing to report.")

    D = np.stack(deltas_xyz, axis=0)  # (N,3)
    dx, dy, dz = D[:, 0], D[:, 1], D[:, 2]
    sx, sy, sz = _axis_stats(dx), _axis_stats(dy), _axis_stats(dz)
    d_arr = np.asarray(dists)

    print("\n" + "=" * 78)
    print(f"EE nearest-point statistics over sequence '{args.seq_name}'")
    print("=" * 78)
    print(f"Frames considered: {start}..{end-1}  (requested {end-start})")
    print(f"Valid frames used : {len(frame_ids)}")
    print(f"Skipped frames    : {skipped_empty} (empty cloud after filters)")
    print(
        f"Filters           : z<= {args.max_depth_m:.3f} m, |x_cam|<= {args.max_x_cam_m:.3f} m"
        + (f", around EE <= {args.around_ee_radius_m:.3f} m" if args.around_ee_radius_m > 0 else "")
    )
    print("-" * 78)
    print("Per-axis delta (nearest_point - ee) [meters]")
    print(f"dx: mean {sx['mean']:.6f}, mean|.| {sx['mean_abs']:.6f}, std {sx['std']:.6f}, rmse {sx['rmse']:.6f}, median {sx['median']:.6f}, p95|.| {sx['p95_abs']:.6f}")
    print(f"dy: mean {sy['mean']:.6f}, mean|.| {sy['mean_abs']:.6f}, std {sy['std']:.6f}, rmse {sy['rmse']:.6f}, median {sy['median']:.6f}, p95|.| {sy['p95_abs']:.6f}")
    print(f"dz: mean {sz['mean']:.6f}, mean|.| {sz['mean_abs']:.6f}, std {sz['std']:.6f}, rmse {sz['rmse']:.6f}, median {sz['median']:.6f}, p95|.| {sz['p95_abs']:.6f}")
    print("-" * 78)
    print("Euclidean nearest distance [meters]")
    print(
        f"mean {np.mean(d_arr):.6f}, std {np.std(d_arr):.6f}, median {np.median(d_arr):.6f}, "
        f"p95 {np.percentile(d_arr, 95.0):.6f}, max {np.max(d_arr):.6f}"
    )
    print("=" * 78 + "\n")

    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # frame, dx, dy, dz, dist
        arr = np.column_stack([np.asarray(frame_ids), D, d_arr])
        header = "frame,dx_m,dy_m,dz_m,dist_m"
        np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
        print(f"Saved per-frame nearest-point deltas to {out_csv}")


if __name__ == "__main__":
    main()

