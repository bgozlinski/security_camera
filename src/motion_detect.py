import cv2
import numpy as np


def highlight_motion_center(frame, fgmask):
    """
    Draw a dot in the center of the detected movement.

    Args:
    - frame: The current frame from the video capture.
    - fgmask: The foreground mask after background subtraction.
    """

    # 5x5 window for noise reduction.
    kernel = np.ones((5, 5), np.uint8)

    # Noise Reduction.
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the fgmask to detect moving objects
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        return cv2.contourArea(max_contour)

    return 0

