"""Send both Franka arms to their home position.

This script mirrors the robot setup used in double_arm_capture.py, but only
initializes the robots and commands them to go to home.
"""

from __future__ import annotations

from pathlib import Path

from crisp_py.robot import Robot


def main() -> None:
    # Initialize robots
    left_arm = Robot(namespace="left")
    right_arm = Robot(namespace="right")
    left_arm.wait_until_ready()
    right_arm.wait_until_ready()

    print("Sending both arms to home position...")
    left_arm.home()
    right_arm.home()
    print("Both arms are now at home.")


if __name__ == "__main__":
    main()

