"""Live test of the braille card detector. Press 's' to save the processing steps, 'q' to quit."""
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CAMERA_URL
from card_detector import detect_card
from undistortion import build_undistort_maps, load_calibration, undistort_frame

# Folder where the processing steps are saved (images for the report)
STEPS_DIR = Path(__file__).resolve().parent.parent / "docs" / "pipeline"


def main():
    K, dist, _ = load_calibration()
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

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
        original = frame.copy()

        letter, corners, steps = detect_card(frame, debug=True)

        # Draw the result: card outline and decoded letter
        if corners is not None:
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            text = f"Letter: {letter}" if letter else "Card found, unknown pattern"
            x, y = corners[0].astype(int)
            cv2.putText(frame, text, (int(x), max(int(y) - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # FPS (smoothed)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Edges", cv2.resize(steps["03_edges"], None, fx=0.5, fy=0.5))
        if "06_card_binary" in steps:
            cv2.imshow("Card (front view)", steps["06_card_binary"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            # Save every step of the pipeline (for the report)
            STEPS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(STEPS_DIR / "00_original.png"), original)
            for name, img in steps.items():
                cv2.imwrite(str(STEPS_DIR / f"{name}.png"), img)
            cv2.imwrite(str(STEPS_DIR / "07_result.png"), frame)
            print("Steps saved in", STEPS_DIR)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()