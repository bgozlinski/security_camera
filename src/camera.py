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
    filename = now.strftime('img_%Y%m%d%H%M%S.jpg')

    # Ensure the "images" directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the frame as an image
    cv2.imwrite(f'images/{filename}', frame)

    return True


def get_camera_record(capture, duration=None):

    if capture is None or not capture.isOpened():
        return False
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    now = datetime.now()
    filename = now.strftime('video_%Y%m%d%H%M%S.mp4')

    if not os.path.exists("records"):
        os.makedirs("records")

    out = cv2.VideoWriter(filename=f'records/{filename}',
                          fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=fps,
                          frameSize=(frame_width, frame_height),
                          )

    start_time = datetime.now()

    try:
        while True:
            frame = get_frame(capture)
            if frame is None:
                break

            out.write(frame)

            if duration and (datetime.now() - start_time).seconds > duration or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    except Exception as e:
        print(f'Error durring recording {e}')
        return False

    finally:
        out.release()
        cv2.destroyAllWindows()

    return True


