"""Live test of the person tracker: put on the yellow vest and walk around.

Green box = vest measured in this frame, red dot = Kalman estimate, red line = trail.
'r' resets the tracker, 'q' quits.
"""
import time
from collections import deque

import cv2
import numpy as np

from camera import CAMERA_URL
from color_mask import load_colors
from person_tracker import PersonTracker
from undistortion import build_undistort_maps, load_calibration, undistort_frame

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    colors = load_colors()
    if "vest" not in colors:
        print("The vest is not in colors.json yet (run hsv_tuner.py and press 1)")
        return

    K, dist, _ = load_calibration()
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

    tracker = PersonTracker(colors["vest"])
    trail = deque(maxlen=40)  # last positions, to draw the path
    map1 = map2 = None
    prev_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break

        # Undistort (maps built only once, with the first frame)
        if map1 is None:
            height, width = frame.shape[:2]
            map1, map2 = build_undistort_maps(K, dist, width, height)
        frame = undistort_frame(frame, map1, map2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = tracker.update(hsv)
        position = tracker.position

        if tracker.box is not None:
            x, y, w, h = tracker.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        if position is None:
            trail.clear()
            status = "No person"
        else:
            trail.append(position)
            cv2.circle(frame, position, 8, RED, -1)
            if tracker.frames_lost == 0:
                status = "Tracking"
            else:
                status = f"Vest hidden for {tracker.frames_lost} frames: predicting"
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, RED, 2)

        # FPS (smoothed)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev_time)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, GREEN, 2)
        cv2.putText(frame, status, (10, 65), FONT, 0.8, WHITE, 2)
        cv2.imshow("Tracker", frame)
        cv2.imshow("Vest mask", cv2.resize(mask, None, fx=0.5, fy=0.5))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset()
            trail.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()