import configparser


def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    camera_port = int(config.get('DEFAULT', 'CameraPort', fallback='0'))
    frame_resize_percent = int(config.get('DEFAULT', 'FrameResizePercent', fallback='50'))
    save_interval_seconds = int(config.get('DEFAULT', 'SaveIntervalSeconds', fallback='10'))
    frame_skip_count = int(config.get('DEFAULT', 'FrameSkip', fallback='1'))
    capture_image_enabled = config['DEFAULT'].getboolean('CaptureImage')
    capture_video_enabled = config['DEFAULT'].getboolean('CaptureVideo')

    motion_area_threshold = int(config.get('MOTION', 'MotionAreaThreshold', fallback='500'))
    dot_radius = int(config.get('MOTION', 'DotRadius', fallback='5'))
    dot_colour = tuple(map(int, config.get('MOTION', 'DotColour', fallback="(0, 0, 255)").strip('()').split(',')))

    return {
        'camera_port': camera_port,
        'frame_resize_percent': frame_resize_percent,
        'save_interval_seconds': save_interval_seconds,
        'frame_skip_count': frame_skip_count,
        'capture_image_enabled': capture_image_enabled,
        'capture_video_enabled': capture_video_enabled,
        'motion_area_threshold': motion_area_threshold,
        'dot_radius': dot_radius,
        'dot_colour': dot_colour
    }

