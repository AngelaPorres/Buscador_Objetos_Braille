"""Live test of the password decoder: show the cards G, U, I, A in order. 'r' resets, 'q' quits."""
import cv2
import numpy as np

from camera import CAMERA_URL
from card_detector import detect_card
from sequence_decoder import SequenceDecoder
from undistortion import build_undistort_maps, load_calibration, undistort_frame

# Text and colour (BGR) shown for each decoder event
MESSAGES = {
    "accepted": ("Correct letter", (0, 200, 0)),
    "wrong": ("Wrong letter: start again", (0, 0, 255)),
    "timeout": ("Too slow: start again", (0, 165, 255)),
    "granted": ("ACCESS GRANTED", (0, 255, 0)),
}
WHITE = (255, 255, 255)


def main():
    K, dist, _ = load_calibration()
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

    decoder = SequenceDecoder()
    map1 = map2 = None
    message, color = "Show the first card", WHITE

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

        letter, corners, _ = detect_card(frame)
        event = decoder.update(letter)
        if event is not None:
            message, color = MESSAGES[event]
            print(event, "".join(decoder.sequence))

        if corners is not None:
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)

        # Password progress, e.g. "G U _ _"
        missing = len(decoder.password) - len(decoder.sequence)
        progress = " ".join(decoder.sequence + ["_"] * missing)
        cv2.putText(frame, f"Password: {progress}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
        cv2.putText(frame, message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        if decoder.sequence and not decoder.granted:
            cv2.putText(frame, f"Time left: {decoder.time_left():.0f} s", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        cv2.imshow("Password", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            decoder.reset()
            message, color = "Reset: show the first card", WHITE

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()