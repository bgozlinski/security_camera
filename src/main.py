import cv2


def get_frame(capture):
    """Capture and return the real frame from the camera."""
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        return None


def threshold_frame(frame):
    """Convert the frame to grayscale, apply Gaussian filter,
       and then use threshold algorithm."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_filter = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, threshold = cv2.threshold(gaussian_filter, 25, 255,
                                 cv2.THRESH_BINARY)
    return threshold


def frame_difference(src1, src2):
    """TODO"""
    diff_frame = cv2.absdiff(src1=src1, src2=src2)
    threshold = threshold_frame(diff_frame)
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

    return capture


def camera_stop(capture):
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = camera_start(port=0)

    prev_frame = None

    while camera.isOpened():
        frame = get_frame(camera)

        if frame is not None:
            if prev_frame is not None:
                threshold = frame_difference(frame, prev_frame)

                cv2.imshow('Raw camera view', frame)
                cv2.imshow('Threshold + Gaussian filter', threshold)

            prev_frame = frame

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break

    camera_stop(camera)


