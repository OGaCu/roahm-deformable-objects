#!/usr/bin/env python3
"""Record RGB + depth to RAM during capture; write disk **after** capture finishes (no robot).

**Default camera: Azure** (`--camera` defaults to `azure`).

**Default outputs:** only `rgbd.npz` (fast path — no MP4/PNG unless you ask). During capture the script only
buffers color + depth in memory, then saves `rgbd.npz` plus small metadata arrays (`elapsed_s`, `n_frames`,
`capture_fps`).

Optional (after capture, slower):
  - `--save-video` → `video.mp4` + `depth.mp4` (JET colormap) at measured FPS
  - `--save-pngs` → `rgb_frames/frame_XXXXXX.png`

Default Azure Kinect `DeviceConfiguration`: 720p BGRA, NFOV_UNBINNED depth, 30 FPS, depth aligned to color
(uint16 mm, 0 = invalid).

Default output directory: `/mnt/mydisk/yellow_bdlo_data/<seq_name>/` (`--datapath` overrides root).

**`--low-memory`:** cannot hold full color buffer — streams **PNG** each frame (and optional MP4 if
`--save-video`) **during** capture; slower capture rate. Prefer default buffered mode + npz-only.

ZED: left RGB + `MEASURE.DEPTH` (meters), ULTRA depth.

**Ctrl+C:** stops capture, then still writes `rgbd.npz` (and optional video/PNGs) for frames collected.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DATAPATH_DEFAULT = "/mnt/mydisk/yellow_bdlo_data"


def _achieved_fps(n_frames: int, elapsed_s: float) -> float:
    if n_frames <= 0 or elapsed_s <= 1e-9:
        return 30.0
    return float(max(1.0, min(120.0, n_frames / elapsed_s)))


def depth_to_jet_bgr(depth_mm: np.ndarray, max_depth_mm: float) -> np.ndarray:
    """uint16 depth in mm (0 = invalid) -> BGR uint8 colormap, same HxW as color."""
    d = depth_mm.astype(np.float32)
    valid = d > 0
    d_norm = np.zeros_like(d, dtype=np.float32)
    d_norm[valid] = np.clip(d[valid] / max_depth_mm, 0.0, 1.0)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)


def depth_meters_to_jet_bgr(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """Float depth in meters (nan/inf/<=0 invalid) -> BGR JET."""
    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    d_norm = np.zeros_like(d, dtype=np.float32)
    d_norm[valid] = np.clip(d[valid] / max_depth_m, 0.0, 1.0)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)


def run_azure(
    out_dir: Path,
    duration_s: float,
    video_fps: float,
    max_depth_mm: float,
    low_memory: bool,
    save_video: bool,
    save_pngs: bool,
) -> None:
    import k4a

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_frames_dir = out_dir / "rgb_frames"
    if low_memory or save_pngs:
        rgb_frames_dir.mkdir(parents=True, exist_ok=True)
        for old in rgb_frames_dir.glob("frame_*.png"):
            old.unlink()

    device = k4a.Device.open()
    if device is None:
        raise RuntimeError(
            "Could not open Azure Kinect device. If the SDK log shows LIBUSB_ERROR_BUSY, "
            "another process already owns the camera (Kinect Viewer, ROS azure_kinect node, "
            "another capture script, Docker, etc.). Quit those completely, then retry."
        )

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
        raise RuntimeError("Azure Kinect failed to start cameras.")

    cal = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    transformation = k4a.Transformation.create(cal)

    for _ in range(5):
        device.get_capture(-1)

    first = device.get_capture(-1)
    if first.color is None:
        device.stop_cameras()
        device.close()
        raise RuntimeError("No color frame from Azure.")
    color0 = cv2.cvtColor(first.color.data, cv2.COLOR_BGRA2BGR)
    h, w = color0.shape[:2]

    if not low_memory:
        est_frames = int(duration_s * 33) + 64
        est_ram_mb = est_frames * h * w * 3 / (1024 * 1024)
        if est_ram_mb > 2048:
            print(
                f"Note: buffered mode may use ~{est_ram_mb:.0f} MB RAM for color frames. "
                f"Use --low-memory if you hit OOM."
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_path = out_dir / "video.mp4"
    depth_path = out_dir / "depth.mp4"

    t_wall_end = time.time() + duration_s
    n_frames = 0
    depth_rows: list[np.ndarray] = []
    color_buffer: list[np.ndarray] = []
    writer_color = writer_depth = None

    if low_memory and save_video:
        writer_color = cv2.VideoWriter(str(color_path), fourcc, video_fps, (w, h))
        writer_depth = cv2.VideoWriter(str(depth_path), fourcc, video_fps, (w, h))
        if not writer_color.isOpened() or not writer_depth.isOpened():
            device.stop_cameras()
            device.close()
            raise RuntimeError(f"Failed to open video writers: {color_path}, {depth_path}")

    if low_memory:
        mode = "low-memory (PNG each frame" + (" + MP4" if save_video else "") + ")"
    else:
        mode = "buffer (disk only after capture" + (
            "; +video" if save_video else ""
        ) + ("; +pngs" if save_pngs else "") + ")"
    print(f"Recording Azure RGB-D to {out_dir} for {duration_s:.1f} s wall-clock [{mode}]...")

    t_cap0 = time.perf_counter()
    t_cap1 = t_cap0
    try:
        try:
            while time.time() < t_wall_end:
                cap = device.get_capture(-1)
                if cap.color is None:
                    continue
                color_bgr = cv2.cvtColor(cap.color.data, cv2.COLOR_BGRA2BGR)
                if cap.depth is not None and transformation is not None:
                    depth_aligned = transformation.depth_image_to_color_camera(cap.depth)
                    depth_data = depth_aligned.data.copy()
                else:
                    depth_data = None

                if depth_data is not None and depth_data.shape[:2] != (h, w):
                    depth_data = cv2.resize(depth_data, (w, h), interpolation=cv2.INTER_NEAREST)

                if low_memory:
                    if writer_color is not None and writer_depth is not None:
                        writer_color.write(color_bgr)
                        if depth_data is not None:
                            writer_depth.write(depth_to_jet_bgr(depth_data, max_depth_mm))
                        else:
                            writer_depth.write(np.zeros((h, w, 3), dtype=np.uint8))
                    if depth_data is not None:
                        depth_rows.append(np.asarray(depth_data, dtype=np.uint16))
                    else:
                        depth_rows.append(np.zeros((h, w), dtype=np.uint16))
                    cv2.imwrite(str(rgb_frames_dir / f"frame_{n_frames:06d}.png"), color_bgr)
                else:
                    color_buffer.append(color_bgr.copy())
                    if depth_data is not None:
                        depth_rows.append(np.asarray(depth_data, dtype=np.uint16))
                    else:
                        depth_rows.append(np.zeros((h, w), dtype=np.uint16))

                n_frames += 1
                if n_frames % 30 == 0:
                    print(f"  frames: {n_frames}")
            t_cap1 = time.perf_counter()
        except KeyboardInterrupt:
            t_cap1 = time.perf_counter()
            print(f"\nStopped early (Ctrl+C) after {n_frames} frames; finalizing...")
    finally:
        if writer_color is not None:
            writer_color.release()
        if writer_depth is not None:
            writer_depth.release()
        device.stop_cameras()
        device.close()

    elapsed = max(t_cap1 - t_cap0, 1e-9)
    achieved = n_frames / elapsed

    if n_frames == 0:
        print("No frames captured; nothing to save.")
        return

    if not low_memory:
        colors = np.stack(color_buffer, axis=0)
    else:
        frame_paths = sorted(rgb_frames_dir.glob("frame_*.png"))
        loaded = [cv2.imread(str(p)) for p in frame_paths]
        if any(x is None for x in loaded):
            raise RuntimeError("Failed to read one or more rgb_frames PNGs for rgbd.npz.")
        colors = np.stack(loaded, axis=0)

    depth_stack = np.stack(depth_rows, axis=0)
    if colors.shape[0] != depth_stack.shape[0]:
        raise RuntimeError(
            f"color ({colors.shape[0]}) vs depth ({depth_stack.shape[0]}) count mismatch."
        )

    rgbd_path = out_dir / "rgbd.npz"
    print(
        f"Capture done: {n_frames} frames in {elapsed:.2f} s wall time → {achieved:.2f} Hz effective."
    )
    np.savez(
        rgbd_path,
        color=colors,
        depth=depth_stack,
        elapsed_s=np.asarray(elapsed, dtype=np.float64),
        n_frames=np.asarray(n_frames, dtype=np.int64),
        capture_fps=np.asarray(achieved, dtype=np.float64),
    )
    print(
        f"Saved {rgbd_path.name}  color {colors.shape}, depth {depth_stack.shape} (uint16 mm)"
    )

    if save_video and not low_memory:
        out_fps = _achieved_fps(n_frames, elapsed)
        print(f"Writing video.mp4 / depth.mp4 at {out_fps:.2f} FPS...")
        writer_color = cv2.VideoWriter(str(color_path), fourcc, out_fps, (w, h))
        writer_depth = cv2.VideoWriter(str(depth_path), fourcc, out_fps, (w, h))
        if not writer_color.isOpened() or not writer_depth.isOpened():
            raise RuntimeError(f"Failed to open video writers: {color_path}, {depth_path}")
        for i, c in enumerate(color_buffer):
            writer_color.write(c)
            writer_depth.write(depth_to_jet_bgr(depth_rows[i], max_depth_mm))
        writer_color.release()
        writer_depth.release()
        print(f"  - {color_path.name}, {depth_path.name}")

    if save_pngs and not low_memory:
        rgb_frames_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(color_buffer):
            cv2.imwrite(str(rgb_frames_dir / f"frame_{i:06d}.png"), c)
        print(f"  - {rgb_frames_dir}/ (frame_*.png)")

    if low_memory and not save_pngs:
        for p in rgb_frames_dir.glob("frame_*.png"):
            p.unlink()

    if low_memory and save_video:
        print(f"  - {color_path.name}, {depth_path.name} (written during capture)")

    print("Done.")


def run_zed(
    out_dir: Path,
    duration_s: float,
    video_fps: float,
    max_depth_m: float,
    low_memory: bool,
    save_video: bool,
    save_pngs: bool,
) -> None:
    import pyzed.sl as sl

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_frames_dir = out_dir / "rgb_frames"
    if low_memory or save_pngs:
        rgb_frames_dir.mkdir(parents=True, exist_ok=True)
        for old in rgb_frames_dir.glob("frame_*.png"):
            old.unlink()

    zed = sl.Camera()
    init = sl.InitParameters()
    init.sdk_verbose = 0
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.camera_fps = int(round(min(30, max(15, video_fps))))
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open ZED camera.")

    rgb_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    for _ in range(10):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError("ZED: could not grab first frame.")

    zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
    frame0 = rgb_mat.get_data()
    h, w = frame0.shape[0], frame0.shape[1]

    if not low_memory:
        est_frames = int(duration_s * 33) + 64
        est_ram_mb = est_frames * h * w * 3 / (1024 * 1024)
        if est_ram_mb > 2048:
            print(
                f"Note: buffered mode may use ~{est_ram_mb:.0f} MB RAM for color frames. "
                f"Use --low-memory if you hit OOM."
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_path = out_dir / "video.mp4"
    depth_path = out_dir / "depth.mp4"

    t_wall_end = time.time() + duration_s
    n_frames = 0
    depth_rows: list[np.ndarray] = []
    color_buffer: list[np.ndarray] = []
    writer_color = writer_depth = None

    if low_memory and save_video:
        writer_color = cv2.VideoWriter(str(color_path), fourcc, video_fps, (w, h))
        writer_depth = cv2.VideoWriter(str(depth_path), fourcc, video_fps, (w, h))
        if not writer_color.isOpened() or not writer_depth.isOpened():
            zed.close()
            raise RuntimeError(f"Failed to open video writers: {color_path}, {depth_path}")

    if low_memory:
        mode = "low-memory (PNG each frame" + (" + MP4" if save_video else "") + ")"
    else:
        mode = "buffer (disk after capture" + ("; +video" if save_video else "") + ("; +pngs" if save_pngs else "") + ")"
    print(f"Recording ZED RGB-D to {out_dir} for {duration_s:.1f} s wall-clock [{mode}]...")

    t_cap0 = time.perf_counter()
    t_cap1 = t_cap0
    try:
        try:
            while time.time() < t_wall_end:
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)

                bgra = rgb_mat.get_data()
                color_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                depth_m = np.asarray(depth_mat.get_data(), dtype=np.float32)

                if depth_m.shape[:2] != color_bgr.shape[:2]:
                    depth_m = cv2.resize(
                        depth_m, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                if low_memory:
                    if writer_color is not None and writer_depth is not None:
                        writer_color.write(color_bgr)
                        writer_depth.write(depth_meters_to_jet_bgr(depth_m, max_depth_m))
                    depth_rows.append(depth_m.copy())
                    cv2.imwrite(str(rgb_frames_dir / f"frame_{n_frames:06d}.png"), color_bgr)
                else:
                    color_buffer.append(color_bgr.copy())
                    depth_rows.append(depth_m.copy())

                n_frames += 1
                if n_frames % 30 == 0:
                    print(f"  frames: {n_frames}")
            t_cap1 = time.perf_counter()
        except KeyboardInterrupt:
            t_cap1 = time.perf_counter()
            print(f"\nStopped early (Ctrl+C) after {n_frames} frames; finalizing...")
    finally:
        if writer_color is not None:
            writer_color.release()
        if writer_depth is not None:
            writer_depth.release()
        zed.close()

    elapsed = max(t_cap1 - t_cap0, 1e-9)
    achieved = n_frames / elapsed

    if n_frames == 0:
        print("No frames captured; nothing to save.")
        return

    if not low_memory:
        colors = np.stack(color_buffer, axis=0)
    else:
        frame_paths = sorted(rgb_frames_dir.glob("frame_*.png"))
        loaded = [cv2.imread(str(p)) for p in frame_paths]
        if any(x is None for x in loaded):
            raise RuntimeError("Failed to read one or more rgb_frames PNGs for rgbd.npz.")
        colors = np.stack(loaded, axis=0)

    depth_stack = np.stack(depth_rows, axis=0)
    if colors.shape[0] != depth_stack.shape[0]:
        raise RuntimeError(
            f"color ({colors.shape[0]}) vs depth ({depth_stack.shape[0]}) count mismatch."
        )

    rgbd_path = out_dir / "rgbd.npz"
    print(
        f"Capture done: {n_frames} frames in {elapsed:.2f} s wall time → {achieved:.2f} Hz effective."
    )
    np.savez(
        rgbd_path,
        color=colors,
        depth=depth_stack,
        elapsed_s=np.asarray(elapsed, dtype=np.float64),
        n_frames=np.asarray(n_frames, dtype=np.int64),
        capture_fps=np.asarray(achieved, dtype=np.float64),
    )
    print(
        f"Saved {rgbd_path.name}  color {colors.shape}, depth {depth_stack.shape} (float32 m depth)"
    )

    if save_video and not low_memory:
        out_fps = _achieved_fps(n_frames, elapsed)
        print(f"Writing video.mp4 / depth.mp4 at {out_fps:.2f} FPS...")
        writer_color = cv2.VideoWriter(str(color_path), fourcc, out_fps, (w, h))
        writer_depth = cv2.VideoWriter(str(depth_path), fourcc, out_fps, (w, h))
        if not writer_color.isOpened() or not writer_depth.isOpened():
            raise RuntimeError(f"Failed to open video writers: {color_path}, {depth_path}")
        for i, c in enumerate(color_buffer):
            writer_color.write(c)
            writer_depth.write(depth_meters_to_jet_bgr(depth_rows[i], max_depth_m))
        writer_color.release()
        writer_depth.release()
        print(f"  - {color_path.name}, {depth_path.name}")

    if save_pngs and not low_memory:
        rgb_frames_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(color_buffer):
            cv2.imwrite(str(rgb_frames_dir / f"frame_{i:06d}.png"), c)
        print(f"  - {rgb_frames_dir}/ (frame_*.png)")

    if low_memory and not save_pngs:
        for p in rgb_frames_dir.glob("frame_*.png"):
            p.unlink()

    if low_memory and save_video:
        print(f"  - {color_path.name}, {depth_path.name} (written during capture)")

    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Capture RGB-D: buffer during capture, then save rgbd.npz (default). Optional MP4/PNGs after."
    )
    p.add_argument(
        "--camera",
        type=str,
        choices=["azure", "zed"],
        default="azure",
        help="Camera backend (default: azure).",
    )
    p.add_argument(
        "--datapath",
        type=str,
        default=DATAPATH_DEFAULT,
        help=f"Root folder for outputs (default: {DATAPATH_DEFAULT}).",
    )
    p.add_argument(
        "--seq-name",
        type=str,
        required=True,
        help="Output folder name: data goes to DATAPATH/<seq-name>/ (default DATAPATH: /mnt/mydisk/yellow_bdlo_data).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording length in seconds (default: 10). Ctrl+C stops early and still saves outputs for frames captured.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="With --low-memory --save-video: MP4 metadata FPS. Otherwise only affects ZED init FPS hint.",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        help="After capture (default buffer mode): write video.mp4 + depth.mp4. With --low-memory: encode during capture.",
    )
    p.add_argument(
        "--save-pngs",
        action="store_true",
        help="After capture (default buffer mode): write rgb_frames/frame_*.png. With --low-memory: keep PNGs (else temp PNGs are deleted after npz).",
    )
    p.add_argument(
        "--low-memory",
        action="store_true",
        help="Spill color as PNG each frame during capture (slower). Use if RAM is too small to buffer all frames.",
    )
    p.add_argument(
        "--max-depth-mm",
        type=float,
        default=1500.0,
        help="Azure: depth colormap max in mm (invalid pixels = black).",
    )
    p.add_argument(
        "--max-depth-m",
        type=float,
        default=1.5,
        help="ZED: depth colormap max in meters.",
    )
    args = p.parse_args()

    out_dir = Path(args.datapath) / args.seq_name

    if args.duration <= 0:
        raise ValueError("--duration must be positive.")

    if args.camera == "azure":
        run_azure(
            out_dir,
            args.duration,
            args.fps,
            args.max_depth_mm,
            args.low_memory,
            args.save_video,
            args.save_pngs,
        )
    else:
        run_zed(
            out_dir,
            args.duration,
            args.fps,
            args.max_depth_m,
            args.low_memory,
            args.save_video,
            args.save_pngs,
        )


if __name__ == "__main__":
    main()
