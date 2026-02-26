"""Compute right-base to left-base transform from the two base2cam calibrations.

Given:
  - T_left_base2cam:  point in left base  -> point in camera
  - T_right_base2cam: point in right base -> point in camera

We compute T_right_base_to_left_base so that:
  p_left = T_right_base_to_left_base @ p_right

Formula: T_right_base_to_left_base = inv(T_left_base2cam) @ T_right_base2cam
"""

import argparse
from pathlib import Path

import numpy as np


def load_base2cam(path: Path) -> np.ndarray:
    """Load 4x4 SE3 from an npz that stores the matrix as arr_0."""
    data = np.load(path)
    T = data["arr_0"]
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    return T


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute right-base to left-base transform from base2cam calibrations."
    )
    parser.add_argument(
        "--left-base2cam",
        type=str,
        default=None,
        help="Path to left base2cam npz (default: DATAPATH/poses/base2cam_transform_left.npz)",
    )
    parser.add_argument(
        "--right-base2cam",
        type=str,
        default=None,
        help="Path to right base2cam npz (default: DATAPATH/poses/base2cam_transform_right.npz)",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects",
        help="Root data path (used when left/right-base2cam not given).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save T_right_base_to_left_base to this npz (default: DATAPATH/poses/right_base_to_left_base.npz)",
    )
    args = parser.parse_args()

    datapath = Path(args.datapath)
    left_path = Path(args.left_base2cam) if args.left_base2cam else datapath / "poses" / "base2cam_transform_left.npz"
    right_path = Path(args.right_base2cam) if args.right_base2cam else datapath / "poses" / "base2cam_transform_right.npz"

    T_left_base2cam = load_base2cam(left_path)
    T_right_base2cam = load_base2cam(right_path)

    # p_cam = T_left_base2cam @ p_left  and  p_cam = T_right_base2cam @ p_right
    # => p_left = inv(T_left_base2cam) @ T_right_base2cam @ p_right
    T_right_to_left = np.linalg.inv(T_left_base2cam) @ T_right_base2cam

    print("T_right_base_to_left_base (4x4):")
    print(T_right_to_left)

    out_path = Path(args.output) if args.output else datapath / "poses" / "right_base_to_left_base.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, T_right_base_to_left_base=T_right_to_left)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
