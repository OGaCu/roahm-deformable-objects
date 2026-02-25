"""Utility helpers for SE(3) waypoint handling and robot control."""

import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class PoseSE3:
    """Pose with position (3,) and orientation as 3x3 rotation matrix."""

    position: np.ndarray  # (3,) in meters
    orientation: np.ndarray  # (3, 3) rotation matrix

    def as_quat(self) -> np.ndarray:
        """Return orientation as quaternion (x, y, z, w)."""
        return Rotation.from_matrix(np.asarray(self.orientation)).as_quat()


def _skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for 3-vector w."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def _se3_log(T: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """SE(3) logarithm: map 4x4 transform to 6-vector (omega, v) in the Lie algebra."""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    w = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(w)
    W = _skew(w)
    if theta < eps:
        V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V_inv = np.eye(3) - 0.5 * W + (1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)) * (W @ W)
    v = V_inv @ p
    return np.hstack([w, v])


def _se3_exp(xi: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """SE(3) exponential: map 6-vector (omega, v) to 4x4 transform."""
    w = xi[0:3]
    v = xi[3:6]
    theta = np.linalg.norm(w)
    W = _skew(w)
    if theta < eps:
        R = np.eye(3) + W + 0.5 * (W @ W)
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        C = (1.0 - A) / (theta * theta)
        R = np.eye(3) + A * W + B * (W @ W)
        V = np.eye(3) + B * W + C * (W @ W)
    p = V @ v
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def _pose_to_matrix(pose: Any) -> np.ndarray:
    """Build 4x4 SE(3) from object with .position and .orientation (3x3 or Rotation)."""
    p = np.asarray(pose.position, dtype=np.float64).reshape(3)
    R = pose.orientation
    if hasattr(R, "as_matrix"):
        R = R.as_matrix()
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def weighted_average_transforms(
    pose1: Any,
    pose2: Any,
    w1: float = 0.5,
    w2: float = 0.5,
) -> PoseSE3:
    """Weighted average of two poses (position + 3x3 orientation) using the Lie group (log/exp).

    Inputs must have .position (3,) and .orientation (3x3 rotation matrix, or scipy Rotation).
    Weights are normalized so that w1 + w2 = 1. The average is computed in the Lie algebra:
    xi_avg = w1*log(T1) + w2*log(T2), then T_avg = exp(xi_avg).

    Args:
        pose1: First pose (object with .position, .orientation).
        pose2: Second pose (object with .position, .orientation).
        w1: Weight for pose1 (default 0.5).
        w2: Weight for pose2 (default 0.5).

    Returns:
        PoseSE3 with .position (3,) and .orientation (3x3) as the weighted average.
    """
    T1 = _pose_to_matrix(pose1)
    T2 = _pose_to_matrix(pose2)
    total = w1 + w2
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    w1, w2 = w1 / total, w2 / total
    xi1 = _se3_log(T1)
    xi2 = _se3_log(T2)
    xi_avg = w1 * xi1 + w2 * xi2
    T_avg = _se3_exp(xi_avg)
    return PoseSE3(
        position=T_avg[0:3, 3].copy(),
        orientation=T_avg[0:3, 0:3].copy(),
    )


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


def move_to_nonblocking(
    robot: Any,
    position: Any = None,
    pose: Any = None,
    speed: float = 0.05,
    done_callback: Callable[[], None] | None = None,
) -> threading.Thread:
    """Run robot.move_to in a background thread; return the thread so you can wait or check when it's done.

    The thread runs robot.move_to(position=position, pose=pose, speed=speed). When the move
    finishes, the thread exits. Use the returned thread to:
    - Wait for completion: thread.join()
    - Check if still moving: thread.is_alive()

    Optionally pass done_callback; it is called (from the worker thread) when move_to returns.

    Args:
        robot: crisp_py Robot instance (must have move_to(position=..., pose=..., speed=...)).
        position: Passed to robot.move_to.
        pose: Passed to robot.move_to.
        speed: Passed to robot.move_to.
        done_callback: Optional callable with no args, invoked when the move completes.

    Returns:
        The started threading.Thread. Thread exits when the move completes.
    """

    def run() -> None:
        try:
            robot.move_to(position=position, pose=pose, speed=speed)
        finally:
            if done_callback is not None:
                done_callback()

    t = threading.Thread(target=run)
    t.start()
    return t
