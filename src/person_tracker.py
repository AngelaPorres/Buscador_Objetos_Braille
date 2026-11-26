"""Person tracker: follows the yellow vest with an HSV mask, morphology and a Kalman filter."""
import cv2
import numpy as np

from color_mask import hsv_mask

MIN_AREA = 1500        # px, smaller yellow blobs are not the vest
GATE = 200             # px, a blob further than this from the prediction is ignored
MAX_LOST = 15          # frames without the vest before the track is dropped (0.5 s at 30 FPS)
PROCESS_NOISE = 1.0    # how much the person may change speed between frames
MEASUREMENT_NOISE = 10.0   # how noisy the measured centre of the vest is

OPEN_KERNEL = np.ones((5, 5), np.uint8)
CLOSE_KERNEL = np.ones((15, 15), np.uint8)  # joins the pieces split by the reflective stripes


def find_vest_blobs(hsv, color_range):
    """Mask of the vest colour and bounding boxes (x, y, w, h) of the big blobs, biggest first."""
    mask = hsv_mask(hsv, color_range["lower"], color_range["upper"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)    # removes small yellow specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)  # fills holes in the vest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    big.sort(key=cv2.contourArea, reverse=True)
    return mask, [cv2.boundingRect(c) for c in big]


def create_kalman(x, y):
    """Constant velocity Kalman filter: state (x, y, vx, vy), measurement (x, y)."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100
    kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kalman


def box_center(box):
    """Centre (x, y) of a box (x, y, w, h)."""
    x, y, w, h = box
    return x + w / 2, y + h / 2


class PersonTracker:
    """Keeps the position of the person between frames, even if the vest is hidden for a moment."""

    def __init__(self, color_range):
        self.color_range = color_range
        self.reset()

    def reset(self):
        self.kalman = None   # None = not tracking anybody
        self.frames_lost = 0
        self.box = None      # vest box measured in the last frame (None if not seen)

    @property
    def position(self):
        """Filtered position (x, y) of the person, or None if not tracking."""
        if self.kalman is None:
            return None
        x, y = self.kalman.statePost[:2, 0]
        return int(x), int(y)

    def update(self, hsv):
        """Process one HSV frame. Returns the vest mask (for debugging)."""
        mask, boxes = find_vest_blobs(hsv, self.color_range)
        self.box = None

        if self.kalman is None:
            # Not tracking: start with the biggest yellow blob
            if boxes:
                self.box = boxes[0]
                self.kalman = create_kalman(*box_center(self.box))
                self.frames_lost = 0
            return mask

        # 1) Predict where the person should be now
        predicted = self.kalman.predict()[:2, 0]

        # 2) The vest is the biggest yellow blob. If it is too far from the prediction it is
        #    a jump (another yellow thing, a bad frame) and it is not used as a measurement
        if boxes and np.hypot(*(np.array(box_center(boxes[0])) - predicted)) < GATE:
            self.box = boxes[0]

        # 3) Correct the prediction with the measurement, or count the frame as lost
        if self.box is not None:
            self.kalman.correct(np.array(box_center(self.box), np.float32).reshape(2, 1))
            self.frames_lost = 0
        else:
            self.frames_lost += 1
            if self.frames_lost > MAX_LOST:
                self.reset()
        return mask