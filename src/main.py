import cv2
import numpy


def get_frame(capture):
    """Capture and return the real frame from the camera."""
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        return None


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

    for _ in range(5):  # Discard first 5 frames, for instance
        get_frame(camera)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg = cv2.createBackgroundSubtractorKNN()

    while camera.isOpened():
        frame = get_frame(camera)

        if frame is not None:
            if prev_frame is not None:
                fgmask = fgbg.apply(frame)
                # Find contours in the thresholded difference image
                contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Iterate over the contours and draw bounding rectangles
                for contour in contours:
                    if cv2.contourArea(contour) < 2500:  # This threshold is arbitrary, you can adjust based on your needs
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('Raw camera view', frame)
                cv2.imshow('fgmask', fgmask)

            prev_frame = frame

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break

    camera_stop(camera)


