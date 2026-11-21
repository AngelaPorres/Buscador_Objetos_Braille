"""Camera calibration loading and real-time lens distortion correction."""
from pathlib import Path

import cv2
import numpy as np

# Resolution of the photos used for calibration (iPhone, 4:3)
CALIB_WIDTH, CALIB_HEIGHT = 4032, 3024

# calibration.npz lives in the project root (one folder above src/)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration.npz"


def load_calibration(path=CALIBRATION_PATH):
    """Load the camera matrix K, the distortion coefficients and the RMS error."""
    data = np.load(path)
    return data["K"], data["dist"], float(data["rms"])


def scale_camera_matrix(K, width, height):
    """Adapt K from the calibration resolution (4032x3024) to the video resolution.

    The 16:9 video is the 4:3 image scaled to the same width and cropped at the
    top and bottom, so fx, fy, cx, cy are scaled and cy is shifted by the crop.
    """
    s = width / CALIB_WIDTH
    K_video = K.astype(np.float64).copy()
    K_video[0, 0] *= s  # fx
    K_video[1, 1] *= s  # fy
    K_video[0, 2] *= s  # cx
    K_video[1, 2] *= s  # cy

    # Pixels cropped at the top (and at the bottom) when going from 4:3 to 16:9
    crop = (CALIB_HEIGHT * s - height) / 2
    K_video[1, 2] -= crop
    return K_video


def build_undistort_maps(K, dist, width, height):
    """Compute the undistortion maps ONCE for the video resolution."""
    K_video = scale_camera_matrix(K, width, height)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_video, dist, None, K_video, (width, height), cv2.CV_16SC2
    )
    return map1, map2


def undistort_frame(frame, map1, map2):
    """Apply the distortion correction to one frame (fast enough for real time)."""
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)