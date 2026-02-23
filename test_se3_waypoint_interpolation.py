"""Quick test and visualization for SE3_waypoint_interpolation."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from helper import SE3_waypoint_interpolation


def make_transform(translation_xyz: np.ndarray, euler_xyz_deg: np.ndarray) -> np.ndarray:
    """Construct a 4x4 transform from translation and XYZ Euler angles (deg)."""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix()
    T[0:3, 3] = translation_xyz
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and visualize SE3 waypoint interpolation.")
    parser.add_argument("--max-step", type=float, default=0.03, help="Max translation step in meters.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="se3_waypoint_interpolation_test.png",
        help="Output image path for waypoint plot.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    args = parser.parse_args()

    start = make_transform(np.array([0.0, 0.0, 0.25]), np.array([0.0, 0.0, 0.0]))
    end = make_transform(np.array([0.25, -0.10, 0.45]), np.array([45.0, -20.0, 60.0]))

    waypoints = SE3_waypoint_interpolation(start, end, max_translation_step=args.max_step)
    positions = np.stack([w[0:3, 3] for w in waypoints], axis=0)
    step_sizes = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.array([])

    if len(step_sizes) > 0:
        max_observed = float(np.max(step_sizes))
        assert max_observed <= args.max_step + 1e-12, (
            f"Observed max step {max_observed:.6f} exceeds threshold {args.max_step:.6f}"
        )

    print(f"Generated {len(waypoints)} waypoints")
    if len(step_sizes) > 0:
        print(f"Max observed translation step: {np.max(step_sizes):.6f} m")
    print(f"Saving visualization to: {args.save_path}")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "o-", linewidth=1.5, markersize=4)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=90, marker="s", label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], s=90, marker="^", label="end")
    ax.set_title("SE(3) Interpolated Waypoints")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(args.save_path, dpi=150)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
