import cv2
import numpy as np
from datetime import datetime, timedelta
from camera import Camera
from motion_detect import MotionDetect
from config.config_loader import load_config


def main():
    """
    Main script to handle security camera operations.
    """
    # Fetching configurations.
    config = load_config('config/config.ini')

    # Start the camera.
    camera = Camera(port=config['camera_port'])

    if not camera.is_opened:
        print("Failed to open camera. Exiting.")
        exit(1)

    # Initialize the Background Subtractor once.
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Create an instance of MotionDetect
    motion_detector = MotionDetect()

    # Set the initial save time to future in save_interval_seconds second.
    next_save_time = datetime.now() + timedelta(seconds=config['save_interval_seconds'])

    # Counter to skip frames for optimization.
    frame_skip = 0

    # Main loop to process each frame.
    while True:
        # Get the current frame.
        frame = camera.capture_frame()

        # Resize frame for better performance.
        frame = camera.resize_frame(frame, scale_percent=config['frame_resize_percent'])

        # Apply the background subtractor to get the foreground mask.
        fg_mask = fgbg.apply(frame)

        # Find motion
        motion_data = motion_detector.find_motion(fg_mask, area_threshold=config['motion_area_threshold'])

        if motion_data:
            motion_area, center_coordinates = motion_data

            # Draw dot on motion
            motion_detector.draw_dot_on_motion(frame, center_coordinates, dot_radius=config['dot_radius'], dot_color=config['dot_colour'])
        else:
            motion_area = False

        # Check for motion based on a defined threshold
        if motion_area:
            current_time = datetime.now()

            if current_time >= next_save_time:
                if config['capture_image_enabled']:
                    camera.capture_image(frame=frame)

                if config['capture_video_enabled']:
                    camera.capture_video(duration=config['save_interval_seconds'])

                next_save_time = current_time + timedelta(seconds=config['save_interval_seconds'])

        # Display the combined frame (original frame + foreground mask).
        combined_frame = np.hstack((frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(winname='frame', mat=combined_frame)

        # Skip some frames for performance optimization.
        frame_skip += 1
        if frame_skip % config['frame_skip_count'] == 0:
            # Break the loop if 'q' is pressed.
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # Release camera and resources
    camera.stop_camera()


if __name__ == "__main__":
    main()
