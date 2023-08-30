import cv2
import numpy as np


class MotionDetect:
    """
    A class to handle motion detection.
    """
    def highlight_motion_center(self, frame, fgmask, dot_radius=5, dot_color=(0, 0, 255), area_threshold=5000):
        """
        Highlights the center of motion in a given frame.

        Args:
            frame (numpy.ndarray): The frame to analyze.
            fgmask (numpy.ndarray): The foreground mask.
            dot_radius (int): The radius of the dot to draw.
            dot_color (tuple): The color of the dot.
            area_threshold (int): The area threshold for motion detection.

        Returns:
            int: The area of the detected motion.
        """
        # Noise Reduction.
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find contours in the fgmask.
        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour.
            max_contour = max(contours, key=cv2.contourArea)

            # If the contour area is above the threshold, draw the dot.
            if cv2.contourArea(max_contour) > area_threshold:
                (x, y, w, h) = cv2.boundingRect(max_contour)
                center_x = int(x + (w / 2))
                center_y = int(y + (h / 2))

                # Draw the dot at the center of the moving object.
                cv2.circle(img=frame,
                           center=(center_x, center_y),
                           radius=dot_radius,
                           color=dot_color,
                           thickness=-1)

                return cv2.contourArea(max_contour)

        return 0
