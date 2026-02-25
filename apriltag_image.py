#!/usr/bin/env python

import os
import cv2
import apriltag

from azure_intrinsics import azure_intrinsics

# Camera intrinsics as (fx, fy, cx, cy) for apriltag.detect_tags
ZED_CAMERA_PARAMS = (716.5634765625, 716.5634765625, 655.4454345703125, 395.7761535644531)

def _camera_params_for(camera):
    """Return (fx, fy, cx, cy) for the given camera."""
    if camera == "zed":
        return ZED_CAMERA_PARAMS
    if camera == "azure":
        K = azure_intrinsics
        return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    raise ValueError(f"Unknown camera: {camera}. Use 'zed' or 'azure'.")

################################################################################
input_image_path = './test_image.png'
output_image_path = './test_image_output.png'
def apriltag_image(input_images=[input_image_path],
                   output_images=False,
                   output_images_path=[output_image_path],
                   display_images=True,
                   detection_window_name='AprilTag',
                   tag_size=0.05,
                   tag_family=None,
                   camera='zed'
                  ):

    '''
    Detect AprilTags from static images.

    Args:   input_images [list(str)]: List of images to run detection algorithm on
            output_images [bool]: Boolean flag to save/not images annotated with detections
            display_images [bool]: Boolean flag to display/not images annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
            camera [str]: 'zed' or 'azure' to select which camera intrinsics to use
    '''
    camera_params = _camera_params_for(camera)

    # parser = ArgumentParser(description='Detect AprilTags from static images.')
    # apriltag.add_arguments(parser)
    # options = parser.parse_args()
    # if tag_family:
    #     options.families = tag_family

    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    # Use default detector options instead of argparse
    options = apriltag.DetectorOptions()
    if tag_family:
        options.families = tag_family

    detector = apriltag.Detector(options=options, searchpath=apriltag._get_dll_path())

    for i, image in enumerate(input_images):

        img = cv2.imread(image)

        print('Reading {}...\n'.format(os.path.split(image)[1]))

        result, overlay = apriltag.detect_tags(img,
                                               detector,
                                               camera_params=camera_params,
                                               tag_size=tag_size,
                                               vizualization=3,
                                               verbose=3,
                                               annotation=True
                                              )

        if output_images:
            output_path = output_images_path[i]
            cv2.imwrite(output_path, overlay)

        if display_images:
            cv2.imshow(detection_window_name, overlay)
            while cv2.waitKey(5) < 0:   # Press any key to load subsequent image
                pass
        
        return result

################################################################################

if __name__ == '__main__':
    apriltag_image()
