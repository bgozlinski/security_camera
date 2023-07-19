import cv2


def get_frame(capture):
    """Capture and return the real frame from the camera."""
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        return None


def otsu_thresholding(frame):
    """Convert the frame to grayscale, apply Gaussian filter,
       and then use Otsu's thresholding algorithm."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_filter = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, threshold = cv2.threshold(gaussian_filter, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    while capture.isOpened():
        frame = get_frame(capture)
        if frame is not None:
            cv2.imshow('Raw camera view', frame)
            threshold = otsu_thresholding(frame)
            cv2.imshow('Otsu + Gaussian filter', threshold)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_start(0)
