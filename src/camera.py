import cv2
import os
from datetime import datetime


class Camera:
    def __init__(self, port):
        self.capture = self.start_camera(port)

    @property
    def is_opened(self):
        return self.capture.isOpened() if self.capture else False

    @property
    def fps(self):
        return int(self.capture.get(cv2.CAP_PROP_FPS)) if self.is_opened else 0

    @property
    def frame_width(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_opened else 0
    @property
    def frame_height(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_opened else 0

    def start_camera(self, port):
        capture = cv2.VideoCapture(port)
        if capture.isOpened() is False:
            print('Error opening the camera')
        return capture

    def stop_camera(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def capture_frame(self):
        ret, frame = self.capture.read()
        return frame if ret else None

    def capture_image(self, frame):
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
            print(f'Error durring recording {e}')
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
