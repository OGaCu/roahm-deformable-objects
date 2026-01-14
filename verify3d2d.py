
import cv2
import numpy as np
from analyze_images_complete_eye_to_hand import t_base2tag

# T_mean is cam to base transformation matrix obtained from previous calibration

T_mean = np.array([[0.0200, 0.9997, -0.0131, 0.0966],
                   [0.2746, -0.0180, -0.9614, 0.2275],
                   [-0.9614, 0.0156, -0.2749, 1.2036],
                   [0.0000, 0.0000,  0.0000, 1.0]])


camera_params = [716.3119506835938, 716.3119506835938, 655.386962890625, 397.7469787597656] # fx, fy, cx, cy
fx = camera_params[0] 
fy = camera_params[1] 
cx = camera_params[2]
cy = camera_params[3]
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# do we have to align base and cam frame again?
R_align = np.array([
    [ 0,  1,  0],   # X_cam =  Y_base
    [ 0,  0, -1],   # Y_cam = -Z_base
    [ 1,  0,  0]    # Z_cam =  X_base
])

max_images = 29

def project_3d_to_2d(point_3d, K, T_cam2base):
    """
    Project a 3D point from base frame to 2D image coordinates.
    
    Args:
        point_3d: 3D point in base frame (3,)
        K: Camera intrinsics matrix (3, 3)
        T_cam2base: Transform from base to camera frame (4, 4)
    
    Returns:
        point_2d: 2D image coordinates (2,) or None if behind camera
        point_cam: 3D point in camera frame
    """
    # Transform point from base frame to camera frame
    point_base_homog = np.hstack([point_3d, 1.0])
    T_base2cam = np.linalg.inv(T_cam2base)
    point_cam = (T_base2cam @ point_base_homog)[:3]
    
    # Check if point is in front of camera (positive Z)
    if point_cam[2] <= 0:
        return None, point_cam
    
    # Project to image using intrinsics
    point_img_homog = K @ point_cam
    point_2d = point_img_homog[:2] / point_img_homog[2]
    
    return point_2d.astype(int), point_cam


print("=== Dry Run: Projecting Tag Center onto Images ===\n")

valid_projections = 0
for i in range(1, max_images + 1):
    img_path = f"./image_pose_{i}.png"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Warning: Could not load {img_path}")
        continue
    
    # Project tag center
    point_2d, point_cam = project_3d_to_2d(t_base2tag[i], K, T_mean)
    
    overlay = img.copy()
    h, w = img.shape[:2]
    
    print(f"Image {i}:")
    print(f"  Tag center in camera frame: {point_cam}")
    
    if point_cam[2] <= 0:
        print("tag behind cam")
    elif point_2d is None:
        print(f"out of bounds")
    else:
        x, y = point_2d
        print(f"  ✓ Projected to 2D: ({x}, {y})")
        # Check if within image bounds
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), 8, (0, 255, 255), -1)      # Cyan filled circle
            cv2.circle(overlay, (x, y), 8, (255, 0, 0), 2)         # Blue outline
            cv2.putText(overlay, f"({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            print(f"Point is WITHIN image bounds")
            valid_projections += 1
        else:
            print(f"Projected point ({x}, {y}) is OUTSIDE image bounds ({w}x{h})")
    
    # Display the image
    cv2.imshow(f"Image {i} - Projected Tag Center", overlay)
    key = cv2.waitKey(500)  # 500ms per image
    
    if key == ord('q'):
        print("\nUser interrupted.")
        break

cv2.destroyAllWindows()
print(f"\n=== Dry Run Complete ===")
print(f"Valid projections: {valid_projections}/{max_images}\n")