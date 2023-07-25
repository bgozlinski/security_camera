import cv2
import os
from datetime import datetime


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


def get_camera_shot(frame):
    """
    Save the provided frame as an image in the "images" folder.
    If the folder doesn't exist, it will be created.

    Args:
    - frame: The frame to be saved.

    Returns:
    - bool: True if the image is saved successfully, False otherwise.
    """

    now = datetime.now()
    filename = now.strftime("img_%Y%m%d%H%M%S")

    # Ensure the "images" directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the frame as an image
    cv2.imwrite(f'images/{filename}.jpg', frame)

    return True
