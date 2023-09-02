import cv2
import numpy as np
from typing import Tuple


class MotionDetect:
    """
    A class to handle motion detection.
    """

    def find_motion(self, fg_mask: np.ndarray, area_threshold: int = 5000) -> Tuple[float, Tuple[int, int]]:
        # Noise Reduction.
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the fgmask.
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour
            max_contour = max(contours, key=cv2.contourArea)

            # If the contour area is above the threshold
            if cv2.contourArea(max_contour) > area_threshold:
                (x, y, w, h) = cv2.boundingRect(max_contour)
                center_x = int(x + (w / 2))
                center_y = int(y + (h / 2))

                return cv2.contourArea(max_contour), (center_x, center_y)

        return None

    def draw_dot_on_motion(self, frame: np.ndarray, center_coordinates: Tuple[int, int], dot_radius: int = 5, dot_color: Tuple[int, int, int] = (0, 0, 255)):
        # Draw the dot at the center of the moving object
        cv2.circle(img=frame,
                   center=center_coordinates,
                   radius=dot_radius,
                   color=dot_color,
                   thickness=-1)
