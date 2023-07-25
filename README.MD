# Movement Detection Camera with Notifications

This project aims to build a movement detection camera that can not only detect movements 
but also notify users about detected movements via SMS, email, or other online communicators. 
The initial implementation primarily focuses on capturing frames and applying filters, with the 
goal of further developing its capabilities to provide the mentioned features.

## Description

The current version of the project uses the OpenCV library to capture video frames from a webcam. 
It also applies Background Subtraction on frames to identify significant changes, 
which can later be interpreted as movement. The primary components include:

- **Real-time Frame Capture**: Captures the real-time frame from the camera.
- **Background Subtractor**: Utilizes MOG2 (Gaussian Mixture-based Background/Foreground Segmentation Algorithm) 
for efficient motion detection.
- **Highlight Motion Center**: Draws a red dot at the center of the detected moving object.

Planned extensions for this project include:

- Integration with SMS gateways to send text messages to users.
- Email notification systems to alert users about detected movements.
- Online communicator notifications for real-time alerts.

## Getting Started

### Prerequisites

  ```bash
  pip install -r requirements.txt