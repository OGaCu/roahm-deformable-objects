"""Extract first frame from each chunk's rgbd.npz and save as PNG.

Usage:
    python extract_first_frames.py
"""

import numpy as np
import cv2
from pathlib import Path

def main():
    base_dir = Path("/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/captured_data_double_arm/dlo1_first400")
    output_dir = base_dir / "first_frames"
    output_dir.mkdir(exist_ok=True)
    
    for i in range(20):
        chunk_dir = base_dir / f"chunk_{i}"
        rgbd_path = chunk_dir / "rgbd.npz"
        
        if not rgbd_path.exists():
            print(f"Warning: {rgbd_path} not found, skipping chunk_{i}")
            continue
        
        # Load rgbd.npz
        rgbd = np.load(rgbd_path)
        
        # Get first frame (index 0) from color array
        # color is (N, H, W, 3) BGR uint8
        colors = rgbd["color"]
        frame0 = colors[0]  # (H, W, 3) BGR
        
        # Save as PNG
        output_path = output_dir / f"frame0_chunk{i}.png"
        cv2.imwrite(str(output_path), frame0)
        print(f"Saved {output_path}")
    
    print(f"\nDone! All first frames saved to {output_dir}")

if __name__ == "__main__":
    main()
