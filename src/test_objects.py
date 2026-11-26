"""Live test of the object detector: draws a box on the label of every object it finds.

Show an object card (C, L, B or E), or press c, l, b or e, to choose the target object.
The "Mask" window shows the mask of the target (or of all the objects if there is no target).
'q' quits.
"""
import time

import cv2
import numpy as np

from camera import CAMERA_URL
from card_detector import detect_card
from color_mask import load_colors
from object_detector import OBJECTS, find_label
from sequence_decoder import LetterStabilizer
from undistortion import build_undistort_maps, load_calibration, undistort_frame

# Colour (BGR) used to draw each object: the same as its printed label
DRAW_COLORS = {"wallet": (40, 20, 220), "keys": (170, 40, 130),
               "bottle": (210, 100, 0), "glasses_case": (60, 160, 0)}
KEYS = {ord(letter.lower()): name for letter, name in OBJECTS.items()}  # c, l, b, e
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    colors = load_colors()
    names = [name for name in OBJECTS.values() if name in colors]
    missing = [name for name in OBJECTS.values() if name not in colors]
    if missing:
        print("Not in colors.json yet (run hsv_tuner.py):", ", ".join(missing))
    if not names:
        return

    K, dist, _ = load_calibration()
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

    stabilizer = LetterStabilizer()
    target = None
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

        # Choose the target object with its card (the password letters are ignored here)
        letter, corners, _ = detect_card(frame)
        confirmed = stabilizer.update(letter)
        if confirmed in OBJECTS and OBJECTS[confirmed] in colors:
            target = OBJECTS[confirmed]
            print("Target:", target)
        if corners is not None:
            cv2.polylines(frame, [corners.astype(np.int32)], True, WHITE, 2)

        # Look for the label of every object
        all_masks = np.zeros(frame.shape[:2], np.uint8)
        target_mask = None
        for name in names:
            mask, label = find_label(hsv, colors[name])
            all_masks = cv2.bitwise_or(all_masks, mask)
            if name == target:
                target_mask = mask
            if label is None:
                continue
            x, y, w, h = label["box"]
            thickness = 4 if name == target else 2
            cv2.rectangle(frame, (x - 6, y - 6), (x + w + 6, y + h + 6), DRAW_COLORS[name], thickness)
            cv2.putText(frame, name, (x, y - 12), FONT, 0.6, DRAW_COLORS[name], 2)

        # FPS (smoothed)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev_time)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Target: {target or '- (show C, L, B or E)'}", (10, 65), FONT, 0.8, WHITE, 2)
        cv2.imshow("Objects", frame)
        shown_mask = target_mask if target_mask is not None else all_masks
        cv2.imshow("Mask", cv2.resize(shown_mask, None, fx=0.5, fy=0.5))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS and KEYS[key] in colors:
            target = KEYS[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()