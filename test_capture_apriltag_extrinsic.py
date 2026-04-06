#!/usr/bin/env python3
"""Capture one Azure image, run AprilTag, print extrinsic translation (x, y, z).

This uses the same local apriltag pipeline (`apriltag_image.py`) used by
`calculate_base_to_cam.py` so behavior stays consistent with your calibration flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apriltag_image import apriltag_image  # noqa: E402


def capture_single_azure_image(output_path: Path) -> Path:
    import k4a

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = k4a.Device.open()
    if device is None:
        raise RuntimeError("Could not open Azure Kinect device.")

    device_config = k4a.DeviceConfiguration(
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_720P,
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
        camera_fps=k4a.EFramesPerSecond.FPS_30,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False,
    )
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        device.close()
        raise RuntimeError("Failed to start Azure cameras.")

    try:
        # Small warm-up before taking one frame.
        for _ in range(5):
            device.get_capture(-1)
        cap = device.get_capture(-1)
        if cap.color is None:
            raise RuntimeError("Failed to capture color frame from Azure.")
        color_bgr = cv2.cvtColor(cap.color.data, cv2.COLOR_BGRA2BGR)
        ok = cv2.imwrite(str(output_path), color_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write image to {output_path}")
    finally:
        device.stop_cameras()
        device.close()

    return output_path


def extract_transform_for_tag(detections, tag_id: int) -> tuple[int, object]:
    """Return (tag_id, 4x4 transform) for the requested tag ID.

    `apriltag_image()` result format mirrors `apriltag.detect_tags(...)` and in this repo is
    consumed as chunks: [detection_obj, 4x4_transform, ...]. We follow that format here.
    """
    if detections is None or len(detections) == 0:
        raise RuntimeError("No AprilTag detections found.")

    # Primary parsing path used in calculate_base_to_cam.py
    for i in range(0, len(detections), 4):
        det = detections[i]
        if hasattr(det, "tag_id") and int(det.tag_id) == tag_id:
            if i + 1 >= len(detections):
                break
            return int(det.tag_id), detections[i + 1]

    # Fallback: if requested tag not present, return first tag with known structure.
    for i in range(0, len(detections), 4):
        det = detections[i]
        if hasattr(det, "tag_id") and (i + 1) < len(detections):
            return int(det.tag_id), detections[i + 1]

    raise RuntimeError("Detections returned, but transform structure was not recognized.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one Azure frame, run AprilTag, print 4x4 extrinsic and xyz translation."
    )
    parser.add_argument(
        "--out-image",
        type=str,
        default=str(THIS_DIR / "test_capture_apriltag_image.png"),
        help="Path to save captured image.",
    )
    parser.add_argument("--tag-id", type=int, default=3, help="Preferred tag id to report (default: 3).")
    parser.add_argument("--tag-size", type=float, default=0.0955, help="AprilTag size in meters (default: 0.093).")
    parser.add_argument("--family", type=str, default="tag36h11", help="Tag family for detector (default: tag36h11).")
    parser.add_argument("--display", action="store_true", help="Display detection overlay window.")
    args = parser.parse_args()

    out_image = capture_single_azure_image(Path(args.out_image))
    print(f"Captured image: {out_image}")

    detections = apriltag_image(
        [str(out_image)],
        output_images=False,
        display_images=args.display,
        tag_size=args.tag_size,
        tag_family=args.family,
        camera="azure",
    )

    found_tag_id, T_cam2tag = extract_transform_for_tag(detections, args.tag_id)
    # Ensure numpy-like indexing works
    T = T_cam2tag
    x = float(T[0, 3])
    y = float(T[1, 3])
    z = float(T[2, 3])

    print("\nAprilTag extrinsic (camera -> tag) 4x4:")
    print(T)
    print(f"\nReported tag id: {found_tag_id}")
    print(f"Translation (x, y, z) [m]: {x:.6f}, {y:.6f}, {z:.6f}")


if __name__ == "__main__":
    main()

