import cv2
import numpy as np
import configparser
from camera import camera_start, camera_stop, capture_frame, capture_image, capture_video, resize_frame
from motion_detect import highlight_motion_center
from datetime import datetime, timedelta

config = configparser.ConfigParser()
config.read('src/config/config.ini')

# Fetching configurations
camera_port = int(config['DEFAULT']['CameraPort'])
frame_resize_percent = int(config['DEFAULT']['FrameResizePercent'])
save_interval_seconds = int(config['DEFAULT']['SaveIntervalSeconds'])
frame_skip_count = int(config['DEFAULT']['FrameSkip'])
capture_image_enabled = config.getboolean('DEFAULT', 'CaptureImage')
capture_video_enabled = config.getboolean('DEFAULT', 'CaptureVideo')
motion_area_threshold = int(config['MOTIION']['MotionAreaThreshold'])
dot_radius = int(config['MOTIION']['DotRadius'])
dot_colour = tuple(map(int, config.get('MOTION', 'DotColour', fallback="(0, 0, 255)").strip('()').split(',')))

if __name__ == "__main__":
    # Start the camera.
    camera, fps = camera_start(port=camera_port)

    # Quick check if camera is available before proceeding.
    if not camera.isOpened():
        print("Camera initialization failed!")
        # Error handling added
        exit()

    # Initialize the Background Subtractor once.
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Set the initial save time to future in save_interval_seconds second.
    next_save_time = datetime.now() + timedelta(seconds=save_interval_seconds)

    # Counter to skip frames for optimization.
    frame_skip = 0

    # User feedback prompt.
    print('Press "q" to stop the video feed')

    # Main loop to process each frame.
    while True:
        # Get the current frame.
        frame = capture_frame(capture=camera)

        # Resize frame for better performance.
        frame = resize_frame(frame=frame,
                             scale_percent=frame_resize_percent)

        # Apply the background subtractor to get the foreground mask.
        fgmask = fgbg.apply(frame)

        # Get contour area.
        motion_area = highlight_motion_center(frame=frame,
                                              fgmask=fgmask,
                                              dot_radius=dot_radius,
                                              dot_color=dot_colour,
                                              area_threshold=motion_area_threshold)

        # Check for motion based on a defined threshold
        if motion_area:
            current_time = datetime.now()

            if current_time >= next_save_time:
                if capture_image_enabled:
                    capture_image(frame=frame)

                if capture_video_enabled:
                    capture_video(capture=camera, duration=save_interval_seconds)

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
    camera_stop(camera)
