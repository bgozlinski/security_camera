import cv2
import numpy as np


def get_frame(capture):
    """
    Capture and return the real frame from the camera.

    Args:
    - capture: The VideoCapture object.

    Returns:
    - frame: Captured frame.
    """

    ret, frame = capture.read()
    if ret:
        return frame
    return None


def camera_start(port):
    """
    Initialize and start the camera.

    Args:
    - port: The camera port (usually 0 for default camera).

    Returns:
    - capture: The VideoCapture object.
    - fps: Frames per second of the capture.
    """

    capture = cv2.VideoCapture(port)

    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print(f'w: {frame_width}\n'
          f'h: {frame_height}\n'
          f'fps: {fps}')

    if capture.isOpened() is False:
        print('Error opening the camera')

    return capture, fps


def camera_stop(capture):
    """
    Release the camera and close any OpenCV windows.

    Args:
    - capture: The VideoCapture object.
    """

    capture.release()
    cv2.destroyAllWindows()


def highlight_motion_center(frame, fgmask):
    """
    Draw a dot on the center of the detected movement.

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


if __name__ == "__main__":
    # Start the camera
    camera, fps = camera_start(port=0)

    # Initialize the Background Subtractor once
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Main loop to process each frame
    while camera.isOpened():
        # Get the current frame
        frame = get_frame(camera)

        # Apply the background subtractor to get the foreground mask
        fgmask = fgbg.apply(frame)

        # Draw dot on detected movement
        highlight_motion_center(frame, fgmask)

        # Display the combined frame (original frame + foreground mask)
        combined_frame = np.hstack((frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('View', combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release camera and resources
    camera_stop(camera)
