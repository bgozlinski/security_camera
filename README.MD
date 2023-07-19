# Movement Detection Camera with Notifications

This project aims to build a movement detection camera that can not only detect movements but also notify users about detected movements via SMS, email, or other online communicators. The initial implementation primarily focuses on capturing frames and applying filters, with the goal of further developing its capabilities to provide the mentioned features.

## Description

The current version of the project uses the OpenCV library to capture video frames from a webcam. It also applies Otsu's thresholding algorithm on the captured frame to identify significant changes, which can later be interpreted as movement. The primary components include:

- **Real-time Frame Capture**: Captures the real-time frame from the camera.
- **Otsu's Thresholding**: The frame is converted to grayscale, followed by the application of a Gaussian filter. Finally, Otsu's thresholding is applied to discern significant changes.

Planned extensions for this project include:

- A movement detection algorithm that intelligently discerns movements from the thresholded frames.
- Integration with SMS gateways to send text messages to users.
- Email notification systems to alert users about detected movements.
- Online communicator notifications for real-time alerts.

## Getting Started

### Prerequisites

- Have Python installed on your machine.
- Install the OpenCV library for Python:

  ```bash
  pip install opencv-python