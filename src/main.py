import cv2
import numpy


def get_frame(capture):
    """Capture and return the real frame from the camera."""
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        return None


def threshold_frame(frame: numpy.ndarray) -> numpy.ndarray:
    """
    Convert the input frame to grayscale, apply Gaussian blur for noise reduction,
    use adaptive thresholding to highlight significant differences in pixel values,
    and then dilate to enhance highlighted regions.

    Parameters:
    - frame: Input image (typically a frame from a video sequence).

    Returns:
    - threshold: Processed binary image highlighting areas of significant change.
    """

    # Convert the frame to grayscale for simpler processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale frame to reduce noise and smooth the image
    gaussian_filter = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply adaptive thresholding using Gaussian method. This helps in dynamically
    # determining the threshold for each pixel based on its neighborhood.
    # The result is inverted (white becomes black and vice versa) using THRESH_BINARY_INV.
    threshold = cv2.adaptiveThreshold(src=gaussian_filter,
                                      maxValue=255,
                                      adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV,
                                      blockSize=11,
                                      C=5)

    # Dilate the thresholded image to enhance and join any fragmented regions
    threshold = cv2.dilate(threshold, None, iterations=2)

    return threshold


def camera_start(port):
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
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera, fps = camera_start(port=0)

    prev_frame = None

    while camera.isOpened():
        print(f'{fps}')
        frame = get_frame(camera)

        if frame is not None:
            if prev_frame is not None:
                diff_frame = cv2.absdiff(src1=frame, src2=prev_frame)
                threshold_diff = threshold_frame(diff_frame)
                cv2.imshow('Raw camera view', frame)
                cv2.imshow('Threshold + Gaussian filter', threshold_diff)

            prev_frame = frame

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break

    camera_stop(camera)


