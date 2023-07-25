import cv2
import numpy as np
from camera import camera_start, camera_stop, get_frame, get_camera_shot
from motion_detect import highlight_motion_center
from datetime import datetime, timedelta


if __name__ == "__main__":
    # Start the camera
    camera, fps = camera_start(port=0)

    # Initialize the Background Subtractor once
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    next_save_time = datetime.now()
    # Main loop to process each frame
    while camera.isOpened():
        # Get the current frame
        frame = get_frame(camera)

        # Apply the background subtractor to get the foreground mask
        fgmask = fgbg.apply(frame)


        # Draw dot on detected movement
        if highlight_motion_center(frame, fgmask):
            current_time = datetime.now()
            if current_time >= next_save_time:
                get_camera_shot(frame=frame)
                next_save_time = current_time + timedelta(seconds=10)

        # Display the combined frame (original frame + foreground mask)
        combined_frame = np.hstack((frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('View', combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release camera and resources
    camera_stop(camera)
