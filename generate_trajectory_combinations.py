"""Generate paired left/right trajectory .task files based on Cartesian distance.

This script:
- Takes two single-arm .task files (e.g. left and right trajectories).
- Parses BOTH the joint angles and cartesian poses from each .task.
- Forms all combinations of poses (i from A, j from B). Right-arm positions are
  transformed to the left base frame using the saved right_base_to_left_base
  transform, then Euclidean distance between left and right (in left base) is computed.
- Keeps only pairs whose distance is BETWEEN a minimum and maximum threshold.
- Writes out TWO new .task files (left and right) that:
    - Have the same number of waypoints.
    - Each index k in the left-out .task corresponds to index k in the right-out .task.
    - Each waypoint is a copy of the original `pose_with_joint_angles` entry so all
      joint and pose data are preserved.

Usage example:

    pixi run python generate_trajectory_combinations.py \\
        --left-task  /path/to/left_traj.task \\
        --right-task /path/to/right_traj.task \\
        --min-distance 0.10 \\
        --max-distance 0.60 \\
        --out-left-task  /tmp/left_traj_filtered.task \\
        --out-right-task /tmp/right_traj_filtered.task
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def _load_task_items(task_path: str | Path) -> list[dict[str, Any]]:
    """Load the `parameter.parameterized_poses` list from a Franka-style .task JSON."""
    task_path = Path(task_path)
    with open(task_path, "r") as f:
        data = json.load(f)
    parameter = data.get("parameter", {})
    items = parameter.get("parameterized_poses", [])
    if not isinstance(items, list):
        raise ValueError(f"'parameter.parameterized_poses' in {task_path} is not a list")
    return items


def _extract_pose_translation(item: dict[str, Any]) -> np.ndarray:
    """Extract the cartesian translation (x,y,z) from a task item.

    The structure is expected to be:
        item["pose_with_joint_angles"]["pose"]  -> list of 16 floats (row-major 4x4)
    or directly:
        item["pose"]  -> list of 16 floats
    We return the translation from the 4x4 transform (column 3).
    """
    pwja = item.get("pose_with_joint_angles") or item
    pose_flat = pwja.get("pose")
    if pose_flat is None or len(pose_flat) != 16:
        raise ValueError("Task item missing 16-element 'pose' array")
    T = np.array(pose_flat, dtype=np.float64).reshape(4, 4).T  # match existing code conventions
    t = T[0:3, 3]
    return t


def _load_right_base_to_left_base(transform_path: Path) -> np.ndarray:
    """Load 4x4 T_right_base_to_left_base from npz (key T_right_base_to_left_base or arr_0)."""
    data = np.load(transform_path)
    if "T_right_base_to_left_base" in data:
        T = data["T_right_base_to_left_base"]
    else:
        T = data["arr_0"]
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    return T


def _transform_points_to_left_base(positions: np.ndarray, T_right_to_left: np.ndarray) -> np.ndarray:
    """Transform (N,3) positions from right base to left base using 4x4 T_right_to_left."""
    N = positions.shape[0]
    ones = np.ones((N, 1), dtype=positions.dtype)
    homog = np.hstack([positions, ones])  # (N, 4)
    return (T_right_to_left @ homog.T).T[:, :3]  # (N, 3)


def _filter_combinations(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    min_distance: float,
    max_distance: float,
    T_right_to_left: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return (i,j) index pairs where distance between left[i] and right[j] is within thresholds.

    If T_right_to_left is provided, right poses are converted to left base before distance computation.
    """
    left_pos = np.stack([_extract_pose_translation(it) for it in left_items], axis=0)  # (N_L, 3)
    right_pos = np.stack([_extract_pose_translation(it) for it in right_items], axis=0)  # (N_R, 3)

    if T_right_to_left is not None:
        right_pos = _transform_points_to_left_base(right_pos, T_right_to_left)  # right in left base

    pairs: list[tuple[int, int]] = []
    for i in range(left_pos.shape[0]):
        for j in range(right_pos.shape[0]):
            d2 = np.sqrt(
                np.square(left_pos[i][0] - right_pos[j][0])
                + np.square(left_pos[i][1] - right_pos[j][1])
            )
            d3 = np.linalg.norm(left_pos[i] - right_pos[j])
            if min_distance <= d2 and d3 <= max_distance:
                print(f"Adding pair ({i}, {j}) with distance d2={d2:.4f} d3={d3:.4f}")
                pairs.append((i, j))
    return pairs


def _build_output_task(
    template_path: str | Path,
    indices: list[int],
    source_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a new .task JSON using template metadata and a subset of items.

    We load the original file, keep all metadata, but replace
    'parameter.parameterized_poses' with the selected subset.
    """
    template_path = Path(template_path)
    with open(template_path, "r") as f:
        data = json.load(f)

    # Create shallow copy of selected items to avoid mutating originals
    new_items = [source_items[i] for i in indices]

    if "parameter" not in data or not isinstance(data["parameter"], dict):
        data["parameter"] = {}
    data["parameter"]["parameterized_poses"] = new_items
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired left/right .task trajectories filtered by Cartesian distance."
    )
    parser.add_argument(
        "--left-task",
        type=str,
        required=True,
        help="Path to left-arm .task file.",
    )
    parser.add_argument(
        "--right-task",
        type=str,
        required=True,
        help="Path to right-arm .task file.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.15, #0.25 for DLO
        help="Minimum allowed Cartesian distance between paired poses (meters).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.46, #0.44 for DLO
        help="Maximum allowed Cartesian distance between paired poses (meters).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of output pairs (keeps first K valid pairs).",
    )
    parser.add_argument(
        "--out-left-task",
        type=str,
        required=True,
        help="Output .task path for filtered left-arm trajectory.",
    )
    parser.add_argument(
        "--out-right-task",
        type=str,
        required=True,
        help="Output .task path for filtered right-arm trajectory.",
    )
    parser.add_argument(
        "--right-to-left-transform",
        type=str,
        default="./poses/right_base_to_left_base.npz",
        help="Path to right_base_to_left_base npz (default: DATAPATH/poses/right_base_to_left_base.npz). "
        "If set, right poses are converted to left base before distance check.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects",
        help="Root path for default right-to-left transform.",
    )

    args = parser.parse_args()

    left_task_path = Path(args.left_task)
    right_task_path = Path(args.right_task)
    datapath = Path(args.datapath)
    transform_path = (
        Path(args.right_to_left_transform)
        if args.right_to_left_transform
        else datapath / "poses" / "right_base_to_left_base.npz"
    )

    T_right_to_left = None
    if transform_path.exists():
        T_right_to_left = _load_right_base_to_left_base(transform_path)
        print(f"Using right_base_to_left_base from {transform_path} (right poses converted to left base for distance).")
    else:
        print(f"No transform at {transform_path}; distances computed in raw frame (left/right may differ in origin).")

    left_items = _load_task_items(left_task_path)
    right_items = _load_task_items(right_task_path)

    print(f"Loaded {len(left_items)} left poses and {len(right_items)} right poses.")

    pairs = _filter_combinations(
        left_items,
        right_items,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        T_right_to_left=T_right_to_left,
    )
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]

    if not pairs:
        print(
            f"No valid pairs found within distance range "
            f"[{args.min_distance:.3f}, {args.max_distance:.3f}] m."
        )
        return

    # Randomize ordering of pairs before writing out, so the resulting trajectories
    # are not sorted by any particular left/right index pattern.
    random.shuffle(pairs)

    left_indices = [i for (i, _) in pairs]
    right_indices = [j for (_, j) in pairs]

    print(f"Selected {len(pairs)} paired waypoints.")

    left_out = _build_output_task(left_task_path, left_indices, left_items)
    right_out = _build_output_task(right_task_path, right_indices, right_items)

    out_left_path = Path(args.out_left_task)
    out_right_path = Path(args.out_right_task)
    out_left_path.parent.mkdir(parents=True, exist_ok=True)
    out_right_path.parent.mkdir(parents=True, exist_ok=True)

    with out_left_path.open("w") as f:
        json.dump(left_out, f, indent=2)
    with out_right_path.open("w") as f:
        json.dump(right_out, f, indent=2)

    print(f"Wrote filtered left task to:  {out_left_path}")
    print(f"Wrote filtered right task to: {out_right_path}")


if __name__ == "__main__":
    main()
