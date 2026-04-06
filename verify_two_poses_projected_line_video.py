#!/usr/bin/env python3
"""
Project the single pose set saved by `single_arm_capture.py` onto captured images.

`single_arm_capture.py` writes, under:
  captured_data_single_arm/<seq_name>/
    rgbd.npz
    single_arm_poses.npz

This script:
  1) Loads the pose file.
  2) Projects each pose's 3D point (x,y,z in base frame) into the camera image using
     the calibrated T_base2cam and camera intrinsics (same math as verify3d2d.py).
  3) Writes **two** MP4s:
     - Default `--output-video`: per frame, the projected point.
     - `--output-video-trajectory`: same overlay plus a polyline trail of all projected points 0..t.
  4) Optionally saves annotated frames under `projected_frames/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    # Make local sibling imports work regardless of the working directory.
    sys.path.insert(0, str(THIS_DIR))

from apriltag_image import _camera_params_for  # noqa: E402


def _load_npz_arr_poses(npz_path: Path) -> np.ndarray:
    """
    Load a positional-saved `np.savez(..., *items)` file where keys are typically arr_0, arr_1, ...
    and each item is a 7-vector: [x, y, z, qx, qy, qz, qw].
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz: {npz_path}")

    data = np.load(npz_path)
    arr_keys = [k for k in data.files if k.startswith("arr_")]
    if not arr_keys:
        # Fallback: take all array-like keys.
        arr_keys = list(data.files)

    def _key_sort(k: str) -> int:
        try:
            return int(k.split("_", 1)[1])
        except Exception:
            return 0

    arr_keys.sort(key=_key_sort)
    vecs = [np.asarray(data[k]) for k in arr_keys]
    if not vecs:
        return np.zeros((0, 7), dtype=np.float64)

    vecs_stacked = np.stack(vecs, axis=0)
    if vecs_stacked.ndim != 2 or vecs_stacked.shape[1] < 3:
        raise ValueError(f"Unexpected pose array shape from {npz_path}: {vecs_stacked.shape}")
    return vecs_stacked


def _load_T_base2cam(npz_path: Path) -> np.ndarray:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing base2cam transform npz: {npz_path}")
    data = np.load(npz_path)

    if "arr_0" in data.files:
        T = np.asarray(data["arr_0"])
    else:
        # Pick the first array.
        first_key = sorted(data.files)[0]
        T = np.asarray(data[first_key])

    if T.shape != (4, 4):
        raise ValueError(f"base2cam transform must be (4,4), got {T.shape} from {npz_path}")
    return T


def project_3d_to_2d(point_3d: np.ndarray, K: np.ndarray, T_base2cam: np.ndarray) -> tuple[float, float] | None:
    """
    Project a single 3D point (in base frame) to image (u, v) pixels.
    Returns None if the point ends up behind the camera (z <= 0 in camera frame).
    """
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(3)
    p_base_h = np.hstack([point_3d, 1.0])  # (4,)
    p_cam = (T_base2cam @ p_base_h)[:3]
    if float(p_cam[2]) <= 0:
        return None

    r_vec, _ = cv2.Rodrigues(T_base2cam[:3, :3].astype(np.float64))
    t_vec = T_base2cam[:3, 3].astype(np.float64)

    # Keep the same call pattern as verify3d2d.py.
    point_3d_cv = point_3d.reshape(1, 1, 3).astype(np.float64)
    point_2d, _ = cv2.projectPoints(point_3d_cv, r_vec, t_vec, K.astype(np.float64), distCoeffs=None)
    uv = point_2d.reshape(2)
    return float(uv[0]), float(uv[1])


def draw_point(img: np.ndarray, pt: tuple[float, float] | None, color_bgr: tuple[int, int, int], radius: int) -> None:
    if pt is None:
        return
    h, w = img.shape[:2]
    x = int(round(pt[0]))
    y = int(round(pt[1]))
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(img, (x, y), radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)


def draw_line(img: np.ndarray, pt1: tuple[float, float] | None, pt2: tuple[float, float] | None, color_bgr, thickness: int) -> None:
    if pt1 is None or pt2 is None:
        return
    h, w = img.shape[:2]
    x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
    x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
    if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
        return
    cv2.line(img, (x1, y1), (x2, y2), color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


def draw_polyline_tail(
    img: np.ndarray,
    pts: list[tuple[float, float] | None],
    end_seg_exclusive: int,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw segments pts[j]→pts[j+1] for j in 0 .. end_seg_exclusive-1 (i.e. trail through pts[0..end_seg_exclusive])."""
    for j in range(end_seg_exclusive):
        if j + 1 >= len(pts):
            break
        draw_line(img, pts[j], pts[j + 1], color_bgr, thickness=thickness)


def combine_images_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img2 is None:
        return img1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h2 != h1:
        new_w2 = int(round(w2 * (h1 / float(h2))))
        img2 = cv2.resize(img2, (new_w2, h1), interpolation=cv2.INTER_AREA)
    return np.concatenate([img1, img2], axis=1)


def combine_images_vertical(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img2 is None:
        return img1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if w2 != w1:
        new_h2 = int(round(h2 * (w1 / float(w2))))
        img2 = cv2.resize(img2, (w1, new_h2), interpolation=cv2.INTER_AREA)
    return np.concatenate([img1, img2], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a single pose stream onto images.")
    default_datapath = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"

    parser.add_argument("--datapath", type=str, default=default_datapath, help="Root hand_to_eye_calibration data directory.")
    parser.add_argument("--seq-name", type=str, default=None, help="Sequence name under captured_data_single_arm/.")
    parser.add_argument("--captured-dir", type=str, default=None, help="Direct path to captured_data_single_arm/<seq>/.")

    parser.add_argument("--camera", type=str, choices=["azure", "zed"], default="azure", help="Camera intrinsics to use.")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="right",
                        help="Which base2cam transform to try by default: base2cam_transform_<side>.npz")
    parser.add_argument(
        "--base2cam-transform",
        type=str,
        default=None,
        help="Path to base2cam_transform npz (recommended). If omitted, auto-searches in --datapath/poses.",
    )

    parser.add_argument("--image-pattern-1", type=str, default="single_arm_image_{i}.png", help="Image filename pattern under frames/.")
    parser.add_argument("--image-pattern-2", type=str, default=None,
                        help="Optional second image filename pattern under frames/. If set, both images are annotated and combined for video.")
    parser.add_argument("--combine", type=str, choices=["side_by_side", "vertical"], default="side_by_side",
                        help="How to combine image 1 and image 2 into one video frame (only used if image-pattern-2 is set).")

    parser.add_argument("--downsample", type=int, default=1, help="Use every Nth pose/image (default 1).")
    parser.add_argument(
        "--frame-delay",
        type=int,
        default=0,
        help="Image index offset for overlay. First pose is drawn on image index=frame-delay.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--output-video", type=str, default="pose_projection.mp4", help="Output mp4 path (relative to captured-dir if not absolute).")
    parser.add_argument(
        "--output-video-trajectory",
        type=str,
        default="pose_projection_trajectory.mp4",
        help="Second mp4: adds a polyline connecting all projected points 0..t (relative to captured-dir if not absolute).",
    )
    parser.add_argument("--trajectory-line-thickness", type=int, default=2, help="Polyline thickness for trajectory video.")
    parser.add_argument("--output-frames-dir", type=str, default=None, help="If set, save annotated frames to this directory (default: projected_frames/ under captured-dir).")

    args = parser.parse_args()

    datapath = Path(args.datapath)
    if args.captured_dir is not None:
        captured_dir = Path(args.captured_dir)
    else:
        if not args.seq_name:
            raise ValueError("Provide either --captured-dir or --seq-name.")
        captured_dir = datapath / "captured_data_single_arm" / args.seq_name

    frames_dir = captured_dir / "frames"
    rgbd_path = captured_dir / "rgbd.npz"
    poses_path = captured_dir / "single_arm_poses.npz"

    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses npz: {poses_path}")

    pose_vecs = _load_npz_arr_poses(poses_path)
    n = pose_vecs.shape[0]
    if n == 0:
        raise RuntimeError("No poses found to project.")

    use_rgbd = rgbd_path.exists()
    colors = None
    if use_rgbd:
        rgbd = np.load(rgbd_path)
        if "color" not in rgbd:
            raise KeyError(f"Missing 'color' in {rgbd_path}")
        colors = np.asarray(rgbd["color"])
        if colors.ndim != 4:
            raise ValueError(f"Unexpected color array shape in {rgbd_path}: {colors.shape}")
    else:
        if not frames_dir.exists():
            raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

    # Intrinsics.
    fx, fy, cx, cy = _camera_params_for(args.camera)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Base->camera transform.
    if args.base2cam_transform:
        T_base2cam_path = Path(args.base2cam_transform)
    else:
        poses_root = datapath / "poses"
        candidates = [
            poses_root / f"base2cam_transform_{args.side}.npz",
            poses_root / "base2cam_transform.npz",
            poses_root / "base2cam_transform_right.npz",
            poses_root / "base2cam_transform_left.npz",
        ]
        T_base2cam_path = next((p for p in candidates if p.exists()), None)

    if T_base2cam_path is None or not Path(T_base2cam_path).exists():
        raise FileNotFoundError(
            "Could not find base2cam transform. Provide --base2cam-transform explicitly, "
            "or ensure one of these exists under "
            f"{datapath / 'poses'}: base2cam_transform_{args.side}.npz / base2cam_transform.npz"
        )

    T_base2cam = _load_T_base2cam(Path(T_base2cam_path))

    downsample = max(1, int(args.downsample))
    pose_indices = list(range(0, n, downsample))

    out_frames_dir = Path(args.output_frames_dir) if args.output_frames_dir else (captured_dir / "projected_frames")
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = Path(args.output_video)
    if not out_video_path.is_absolute():
        out_video_path = captured_dir / out_video_path

    out_video_traj_path = Path(args.output_video_trajectory)
    if not out_video_traj_path.is_absolute():
        out_video_traj_path = captured_dir / out_video_traj_path

    # Precompute 2D projections for every index we step through (for trajectory polylines).
    uv_list: list[tuple[float, float] | None] = []
    for i in pose_indices:
        uv_list.append(project_3d_to_2d(pose_vecs[i, 0:3], K, T_base2cam))

    writer = None
    writer_traj = None
    wrote_frames = 0

    # Output styling.
    color_point = (0, 255, 0)    # green
    traj_color = (0, 180, 0)     # darker green trail
    radius = 4
    traj_thick = max(1, int(args.trajectory_line_thickness))

    for step_idx, i in enumerate(pose_indices):
        img_idx = i + int(args.frame_delay)
        if img_idx < 0:
            # Skip overlays that would map to negative image indices.
            continue
        if use_rgbd:
            if img_idx >= colors.shape[0]:
                print(f"Skipping missing rgbd frame index: {img_idx}")
                continue
            img1 = colors[img_idx]
        else:
            img1_path = frames_dir / args.image_pattern_1.format(i=img_idx)
            img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
            if img1 is None:
                print(f"Skipping missing image: {img1_path}")
                continue

        # 2nd image is optional; if missing we fall back to image1.
        img2 = None
        if args.image_pattern_2 and not use_rgbd:
            img2_path = frames_dir / args.image_pattern_2.format(i=img_idx)
            img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
            if img2 is None:
                print(f"Warning: missing image-pattern-2 at {img2_path}; using image-pattern-1 for both.")
                img2 = None

        uv = uv_list[step_idx]

        def annotate(img: np.ndarray, *, with_trajectory: bool) -> np.ndarray:
            out = img.copy()
            if with_trajectory:
                # Connect all previous projected points along each pose stream up to current index.
                if step_idx > 0:
                    draw_polyline_tail(out, uv_list, step_idx, traj_color, traj_thick)
            draw_point(out, uv, color_point, radius=radius)
            return out

        img1_ann = annotate(img1, with_trajectory=False)
        img2_ann = annotate(img2, with_trajectory=False) if img2 is not None else None

        img1_traj = annotate(img1, with_trajectory=True)
        img2_traj = annotate(img2, with_trajectory=True) if img2 is not None else None

        if img2_ann is None:
            canvas = img1_ann
            canvas_traj = img1_traj
        else:
            canvas = (
                combine_images_side_by_side(img1_ann, img2_ann)
                if args.combine == "side_by_side"
                else combine_images_vertical(img1_ann, img2_ann)
            )
            canvas_traj = (
                combine_images_side_by_side(img1_traj, img2_traj)
                if args.combine == "side_by_side"
                else combine_images_vertical(img1_traj, img2_traj)
            )

        # Write output frame (connector-only overlay for PNG sequence).
        out_frame_path = out_frames_dir / f"projected_{step_idx:06d}_pose{i}_img{img_idx}.png"
        cv2.imwrite(str(out_frame_path), canvas)

        if writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, float(args.fps), (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_video_path}")
            ht, wt = canvas_traj.shape[:2]
            if (ht, wt) != (h, w):
                raise RuntimeError(f"Trajectory canvas size {(wt, ht)} != main canvas {(w, h)}")
            writer_traj = cv2.VideoWriter(str(out_video_traj_path), fourcc, float(args.fps), (w, h))
            if not writer_traj.isOpened():
                writer.release()
                raise RuntimeError(f"Failed to open trajectory video writer: {out_video_traj_path}")

        writer.write(canvas)
        writer_traj.write(canvas_traj)
        wrote_frames += 1

    if writer is not None:
        writer.release()
    if writer_traj is not None:
        writer_traj.release()

    print(f"Done. Wrote {wrote_frames} frames to video: {out_video_path}")
    print(f"Done. Wrote {wrote_frames} frames to trajectory video: {out_video_traj_path}")
    print(f"Annotated frames saved under: {out_frames_dir}")


if __name__ == "__main__":
    main()
