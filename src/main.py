import cv2
import numpy as np
import configparser
from datetime import datetime, timedelta
from camera import Camera
from motion_detect import MotionDetect

config = configparser.ConfigParser()
config.read('config/config.ini')


# Fetching configurations
camera_port = int(config.get('DEFAULT', 'CameraPort', fallback='0'))
frame_resize_percent = int(config.get('DEFAULT', 'FrameResizePercent', fallback='50'))
save_interval_seconds = int(config.get('DEFAULT', 'SaveIntervalSeconds', fallback='10'))
frame_skip_count = int(config.get('DEFAULT', 'FrameSkip', fallback='1'))
capture_image_enabled = config['DEFAULT'].getboolean('CaptureImage')
capture_video_enabled = config['DEFAULT'].getboolean('CaptureVideo')

motion_area_threshold = int(config.get('MOTION', 'MotionAreaThreshold', fallback='500'))
dot_radius = int(config.get('MOTION', 'DotRadius', fallback='5'))
dot_colour = tuple(map(int, config.get('MOTION', 'DotColour', fallback="(0, 0, 255)").strip('()').split(',')))

if __name__ == "__main__":
    # Start the camera.
    camera = Camera(port=camera_port)

    if not camera.is_opened:
        print("Failed to open camera. Exiting.")
        exit(1)

    # Initialize the Background Subtractor once.
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Create an instance of MotionDetect
    motion_detector = MotionDetect()

    # Set the initial save time to future in save_interval_seconds second.
    next_save_time = datetime.now() + timedelta(seconds=save_interval_seconds)

    # Counter to skip frames for optimization.
    frame_skip = 0

    # Main loop to process each frame.
    while True:
        # Get the current frame.
        frame = camera.capture_frame()

        # Resize frame for better performance.
        frame = camera.resize_frame(frame, scale_percent=frame_resize_percent)

        # Apply the background subtractor to get the foreground mask.
        fgmask = fgbg.apply(frame)

        # Get contour area.
        motion_area = motion_detector.highlight_motion_center(frame=frame,
                                                              fgmask=fgmask,
                                                              dot_radius=dot_radius,
                                                              dot_color=dot_colour,
                                                              area_threshold=motion_area_threshold)

        # Check for motion based on a defined threshold
        if motion_area:
            current_time = datetime.now()

            if current_time >= next_save_time:
                if capture_image_enabled:
                    camera.capture_image(frame=frame)

                if capture_video_enabled:
                    camera.capture_video(duration=save_interval_seconds)

                next_save_time = current_time + timedelta(seconds=save_interval_seconds)

        # Display the combined frame (original frame + foreground mask).
        combined_frame = np.hstack((frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(winname='frame', mat=combined_frame)

        # Skip some frames for performance optimization.
        frame_skip += 1
        if frame_skip % frame_skip_count == 0:
            # Break the loop if 'q' is pressed.
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # Release camera and resources
    camera.stop_camera()
