"""Utility helpers for SE(3) waypoint handling."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def SE3_waypoint_interpolation(
    start_waypoint: np.ndarray,
    end_waypoint: np.ndarray,
    max_translation_step: float = 0.03,
) -> list[np.ndarray]:
    """Interpolate SE(3) transforms so adjacent translations are bounded.

    Args:
        start_waypoint: 4x4 start transform.
        end_waypoint: 4x4 end transform.
        max_translation_step: Max Euclidean translation [m] between consecutive waypoints.

    Returns:
        List of 4x4 transforms including start and end.
    """
    if start_waypoint.shape != (4, 4) or end_waypoint.shape != (4, 4):
        raise ValueError("Both waypoints must be 4x4 SE(3) matrices.")
    if max_translation_step <= 0:
        raise ValueError("max_translation_step must be > 0.")

    p0 = start_waypoint[0:3, 3]
    p1 = end_waypoint[0:3, 3]
    distance = np.linalg.norm(p1 - p0)

    num_segments = max(1, int(np.ceil(distance / max_translation_step)))
    alphas = np.linspace(0.0, 1.0, num_segments + 1)

    r0 = Rotation.from_matrix(start_waypoint[0:3, 0:3])
    r1 = Rotation.from_matrix(end_waypoint[0:3, 0:3])
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0.as_matrix(), r1.as_matrix()])))

    interpolated_waypoints = []
    for a in alphas:
        T = np.eye(4, dtype=np.float64)
        T[0:3, 3] = (1.0 - a) * p0 + a * p1
        T[0:3, 0:3] = slerp([a]).as_matrix()[0]
        interpolated_waypoints.append(T)

    return interpolated_waypoints


def densify_waypoints(
    raw_waypoints: list[np.ndarray],
    max_translation_step: float = 0.03,
) -> list[np.ndarray]:
    """Expand a waypoint list so adjacent translation steps are bounded."""
    if len(raw_waypoints) == 0:
        return []
    if len(raw_waypoints) == 1:
        return raw_waypoints.copy()

    waypoints: list[np.ndarray] = []
    for idx in range(len(raw_waypoints) - 1):
        segment = SE3_waypoint_interpolation(
            raw_waypoints[idx],
            raw_waypoints[idx + 1],
            max_translation_step=max_translation_step,
        )
        if idx == 0:
            waypoints.extend(segment)
        else:
            # Skip duplicated boundary waypoint between consecutive segments.
            waypoints.extend(segment[1:])

    return waypoints
