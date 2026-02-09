import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.sdk_verbose = 1
init_params.camera_resolution = sl.RESOLUTION.AUTO
init_params.camera_fps = 30
zed.open(init_params)

print(f"Serial Number: {zed.get_camera_information().serial_number}")

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
# Focal length of the left eye in pixels
focal_left_x = calibration_params.left_cam.fx
focal_left_y = calibration_params.left_cam.fy
cx = calibration_params.left_cam.cx
cy = calibration_params.left_cam.cy
# First radial distortion coefficient
ks = calibration_params.left_cam.disto
# Translation between left and right eye on x-axis
tx = calibration_params.stereo_transform.get_translation().get()[0]
# Horizontal field of view of the left eye in degrees
h_fov = calibration_params.leftcam.h_fov
print(f"fx: {focal_left_x}, fy: {focal_left_y}, cx: {cx} {cy=} {ks}")
zed.close()