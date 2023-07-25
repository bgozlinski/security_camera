import cv2
from datetime import datetime


def highlight_motion_center(frame, fgmask):
    """
    Draw a dot in the center of the detected movement.

    Args:
    - frame: The current frame from the video capture.
    - fgmask: The foreground mask after background subtraction.
    """

    # Find contours in the fgmask to detect moving objects
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # If the contour area is above a threshold, draw a red dot
        if cv2.contourArea(max_contour) > 5000:
            (x, y, w, h) = cv2.boundingRect(max_contour)
            center_x = int(x + (w / 2))
            center_y = int(y + (h / 2))

            # Draw a red dot (circle) at the center of the moving object
            cv2.circle(img=frame,
                       center=(center_x, center_y),
                       radius=5,
                       color=(0, 0, 255),
                       thickness=-1)
    return True

