import cv2
import os
import logging
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)


class CameraError(Exception):
    pass


@contextmanager
def camera_context(port):
    camera = Camera(port)
    try:
        yield camera
    finally:
        camera.stop_camera()


class Camera:
    """
    A class to handle camera operations.
    """
    def __init__(self, port):
        """
        Initialize the Camera object.

        Args:
            port (int): The port number of the camera.
        """
        self.capture = self.start_camera(port)

    @property
    def is_opened(self):
        """
        Check if the camera is opened.

        Returns:
            bool: True if the camera is opened, False otherwise.
        """
        return self.capture.isOpened() if self.capture else False

    @property
    def fps(self):
        """
        Get the frames per second (FPS) of the camera.

        Returns:
            int: The FPS of the camera.
        """
        return int(self.capture.get(cv2.CAP_PROP_FPS)) if self.is_opened else 0

    @property
    def frame_width(self):
        """
        Get the frame width of the camera.

        Returns:
            int: The frame width.
        """
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_opened else 0

    @property
    def frame_height(self):
        """
        Get the frame height of the camera.

        Returns:
            int: The frame height.
        """
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_opened else 0

    def start_camera(self, port):
        """
        Start the camera and return the capture object.

        Args:
            port (int): The port number of the camera.

        Returns:
            cv2.VideoCapture: The capture object for the camera.
        """
        capture = cv2.VideoCapture(port)
        if not capture.isOpened():
            logging.error('Error opening the camera')
            raise CameraError('Could not open camera')
        return capture

    def stop_camera(self):
        """
        Stop the camera and release all resources.
        """
        self.capture.release()
        cv2.destroyAllWindows()

    def capture_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame.
        """
        ret, frame = self.capture.read()
        return frame if ret else None

    def capture_image(self, frame):
        """
        Capture an image and save it.

        Args:
            frame (numpy.ndarray): The frame to save as an image.

        Returns:
            bool: True if the image was successfully saved, False otherwise.
        """
        try:
            now = datetime.now()
            filename = now.strftime('img_%Y%m%d%H%M%S.jpg')

            # Ensure the "images" directory exists.
            if not os.path.exists("../images"):
                os.makedirs("../images")

            # Put current DateTime on frame
            font = cv2.FONT_HERSHEY_PLAIN
            time_stamp = now.strftime('%Y-%m-%d %H:%M:%S')

            cv2.putText(frame, str(time_stamp), (0, 10),
                        font, 1, (255, 0, 0), 1, cv2.LINE_AA)

            # Save the frame as an image.
            cv2.imwrite(f'../images/{filename}', frame)
            print(f"Image saved as images/{filename}")  # Add this line to confirm the image is saved

            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def capture_video(self, duration=None):
        """
        Capture video for a specified duration.

        Args:
            duration (int, optional): The duration for which to capture video.

        Returns:
            bool: True if the video was successfully captured, False otherwise.
        """
        if self.capture is None or not self.capture.isOpened():
            return False

        frame_width = self.frame_width
        frame_height = self.frame_height
        fps = self.fps

        now = datetime.now()
        filename = now.strftime('video_%Y%m%d%H%M%S.mp4')

        # Ensure the "records" directory exists
        if not os.path.exists("../records"):
            os.makedirs("../records")

        # Initialize video writer
        out = cv2.VideoWriter(filename=f'../records/{filename}',
                              fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                              fps=fps,
                              frameSize=(frame_width, frame_height),
                              )

        start_time = datetime.now()

        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    break

                # Put current DateTime on each frame
                font = cv2.FONT_HERSHEY_PLAIN
                now = datetime.now()
                time_stamp = now.strftime('%Y-%m-%d %H:%M:%S')

                cv2.putText(frame, str(time_stamp), (0, 40),
                            font, 2, (255, 0, 0), 1, cv2.LINE_AA)

                # Display the resulting frame
                cv2.imshow('View', frame)
                out.write(frame)

                if duration and (datetime.now() - start_time).seconds > duration or (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

        except Exception as e:
            print(f'Error during recording {e}')
            return False

        finally:
            out.release()
            cv2.destroyAllWindows()

        return True

    @staticmethod
    def resize_frame(frame, scale_percent=50):
        """
        Resizes a given frame by the specified percentage.

        Args:
        - frame (numpy.ndarray): The input frame to be resized.
        - scale_percent (int, optional): The percentage by which the frame should be resized. Default is 50%.

        Returns:
        - numpy.ndarray: The resized frame.
        """

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        return cv2.resize(frame, (width, height))
