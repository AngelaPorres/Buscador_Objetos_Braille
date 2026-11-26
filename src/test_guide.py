"""Live test of the hot/cold guide with voice.
 
Put on the vest, show an object card (C, L, B or E) or press c, l, b or e, and walk:
the voice says if you are getting hotter or colder. The distance on screen helps to
tune FOUND_DISTANCE and BURNING_DISTANCE in guide.py. 'r' starts again, 'q' quits.
"""
import time
 
import cv2
import numpy as np
 
from camera import CAMERA_URL
from card_detector import detect_card
from color_mask import load_colors
from guide import HotColdGuide, ObjectLocator, phrase
from object_detector import OBJECTS, find_label
from person_tracker import PersonTracker
from sequence_decoder import LetterStabilizer
from speaker import Speaker
from undistortion import build_undistort_maps, load_calibration, undistort_frame
 
# Colour (BGR) of the line person-object after each hint
HINT_COLORS = {"hotter": (0, 0, 255), "burning": (0, 0, 255), "colder": (255, 128, 0),
               "found": (0, 255, 0), "lost": (128, 128, 128)}
KEYS = {ord(letter.lower()): name for letter, name in OBJECTS.items()}  # c, l, b, e
WHITE = (255, 255, 255)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
 
 
def main():
    colors = load_colors()
    missing = [name for name in ["vest", *OBJECTS.values()] if name not in colors]
    if missing:
        print("Not in colors.json yet (run hsv_tuner.py):", ", ".join(missing))
        if "vest" in missing:
            return
 
    K, dist, _ = load_calibration()
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return
 
    speaker = Speaker()
    tracker = PersonTracker(colors["vest"])
    stabilizer = LetterStabilizer()
    guide = HotColdGuide()
    locator = ObjectLocator()
    target = None
    last_hint = None
    map1 = map2 = None
    prev_time = time.perf_counter()
    fps = 0.0
 
    def start_search(name):
        nonlocal target, locator, last_hint
        target, locator, last_hint = name, ObjectLocator(), None
        guide.reset()
        speaker.say(phrase("start", name))
        print("Target:", name)
 
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
 
        tracker.update(hsv)
        person = tracker.position
 
        if target is None:
            # Waiting for an object card (cards are only searched while there is no target)
            letter, corners, _ = detect_card(frame)
            confirmed = stabilizer.update(letter)
            if confirmed in OBJECTS and OBJECTS[confirmed] in colors:
                start_search(OBJECTS[confirmed])
            if corners is not None:
                cv2.polylines(frame, [corners.astype(np.int32)], True, WHITE, 2)
        else:
            # Searching: locate the object and give a hint when it is time
            _, label = find_label(hsv, colors[target])
            locator.update(label)
            event = guide.update(person, locator.position, time.monotonic())
            if event is not None:
                last_hint = event
                speaker.say(phrase(event, target))
                print("Hint:", event, f"(distance {guide.distance or 0:.0f} px)")
 
        # Drawing
        if person is not None:
            cv2.circle(frame, person, 8, RED, -1)
        if target is not None and locator.position is not None:
            cv2.circle(frame, locator.position, 14, WHITE, 2)
            if person is not None:
                cv2.line(frame, person, locator.position, HINT_COLORS.get(last_hint, WHITE), 2)
 
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev_time)
        prev_time = now
 
        if target is None:
            status = "Show an object card (C, L, B or E)"
        elif guide.found:
            status = f"FOUND: {target}"
        elif locator.position is None:
            status = f"Looking for the label of the {target}..."
        else:
            distance = "-" if guide.distance is None else f"{guide.distance:.0f} px"
            status = f"Target: {target} | distance: {distance} | last hint: {last_hint or '-'}"
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 65), FONT, 0.8, WHITE, 2)
        cv2.imshow("Guide", frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            target, last_hint = None, None
            guide.reset()
        if key in KEYS and KEYS[key] in colors:
            start_search(KEYS[key])
 
    speaker.close()
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()