import cv2
import numpy as np


def highlight_motion_center(fgmask):
    """
    Highlights the center of detected motion in the given frame using a red dot.

    This function works by finding contours in the foreground mask, determining
    the center of the largest contour (representing the most significant motion),
    and then drawing a red dot on the original frame at that center location.

    Args:
    - frame (numpy.ndarray): The current frame from the video capture.
    - fgmask (numpy.ndarray): The foreground mask after background subtraction, used to detect motion.

    Returns:
    - bool: True if a notable motion center is detected and highlighted, False otherwise.
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

