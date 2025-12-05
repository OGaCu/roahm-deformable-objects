import pyzed.sl as sl
import cv2
import numpy as np
import apriltag
from apriltag_image import apriltag_image

# def undistort_image(img_bgr, fx, fy, cx, cy, dist_coeffs):
#     """
#     dist_coeffs: iterable of 4–8 values for the standard OpenCV model
#                  [k1, k2, p1, p2, k3, k4, k5, k6] (extra terms optional)
#     """
#     h, w = img_bgr.shape[:2]
#     K = np.array([[fx,  0, cx],
#                   [ 0, fy, cy],
#                   [ 0,  0,  1]], dtype=np.float64)
#     D = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)

#     # Fast + minimal:
#     newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)  # alpha=0: crop all black edges
#     undistorted = cv2.undistort(img_bgr, K, D, None, newK)
#     return undistorted

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.sdk_verbose = 1
init_params.camera_resolution = sl.RESOLUTION.AUTO
init_params.camera_fps = 30
zed.open(init_params)

print(f"Serial Number: {zed.get_camera_information().serial_number}")

image = sl.Mat()
runtime_params = sl.RuntimeParameters()
pose_count = 1
while(pose_count <= 10):
    while(input() != "n"):
        pass
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT) # can we get a combined mixed view?
        frame = image.get_data()
        cv2.imwrite(f"image_pose_{pose_count}.png", frame)
        print(f"Image Captured {pose_count}")
        # detections = apriltag_image([f"./image_pose_{pose_count}.png"], output_images=True, display_images=True)
        # print("Detections: ", detections[1])
        pose_count += 1



else:
    print("Failure: Zed failed to capture image")

# calibration__params = zed.get_camera_information().camera_configuration.calibration_parameters
# # Focal length of the left eye in pixels
# focal_left_x = calibration_params.left_cam.fx
# focal_left_y = calibration_params.left_cam.fy
# cx = calibration_params.left_cam.cx
# cy = calibration_params.left_cam.cy
# # First radial distortion coefficient
# ks = calibration_params.left_cam.disto
# # Translation between left and right eye on x-axis
# tx = calibration_params.stereo_transform.get_translation().get()[0]
# # Horizontal field of view of the left eye in degrees
# h_fov = calibration_params.leftcam.h_fov
# print(f"fx: {focal_left_x}, fy: {focal_left_y}, cx: {cx} {cy=} {ks}")
zed.close()