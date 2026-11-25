"""HSV colour tuner with trackbars (like the interface of Lab 1).

Click on an object to centre the ranges on its colour, fine-tune the trackbars and
press a number to save: 1 vest, 2 wallet, 3 keys, 4 bottle, 5 glasses case. 'q' quits.
"""
import cv2
import numpy as np

from camera import CAMERA_URL
from color_mask import hsv_mask, load_colors, save_color

WINDOW = "HSV tuner"
SCALE = 0.5  # the window shows the images at half size
NAMES = {ord("1"): "vest", ord("2"): "wallet", ord("3"): "keys",
         ord("4"): "bottle", ord("5"): "glasses_case"}
TRACKBARS = [("H min", 179), ("H max", 179), ("S min", 255),
             ("S max", 255), ("V min", 255), ("V max", 255)]

state = {"hsv": None}  # last HSV frame, used by the mouse callback


def set_ranges(lower, upper):
    """Move the trackbars to the given range."""
    values = [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]
    for (name, _), value in zip(TRACKBARS, values):
        cv2.setTrackbarPos(name, WINDOW, int(value))


def get_ranges():
    """Read the current range from the trackbars."""
    v = [cv2.getTrackbarPos(name, WINDOW) for name, _ in TRACKBARS]
    return (v[0], v[2], v[4]), (v[1], v[3], v[5])


def on_mouse(event, x, y, flags, param):
    """Clicking on the image centres the range on the colour of the clicked point."""
    hsv = state["hsv"]
    if event != cv2.EVENT_LBUTTONDOWN or hsv is None:
        return
    panel_width = hsv.shape[1] * SCALE
    x = int((x % panel_width) / SCALE)  # works on both halves of the window
    y = int(y / SCALE)
    patch = hsv[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].reshape(-1, 3)
    h, s, v = np.median(patch, axis=0).astype(int)
    print(f"Clicked HSV = ({h}, {s}, {v})")
    # +-10 in hue (the % 180 handles red, which wraps around), generous S and V
    set_ranges(((h - 10) % 180, max(s - 70, 0), max(v - 70, 0)), ((h + 10) % 180, 255, 255))


def main():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

    cv2.namedWindow(WINDOW)
    for name, maximum in TRACKBARS:
        cv2.createTrackbar(name, WINDOW, 0, maximum, lambda _: None)
    set_ranges((0, 80, 80), (179, 255, 255))
    cv2.setMouseCallback(WINDOW, on_mouse)
    saved = load_colors()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        state["hsv"] = hsv
        lower, upper = get_ranges()
        mask = hsv_mask(hsv, lower, upper)
        segmented = cv2.bitwise_and(frame, frame, mask=mask)

        # Original and segmented image side by side
        view = cv2.resize(np.hstack([frame, segmented]), None, fx=SCALE, fy=SCALE)
        cv2.putText(view, "Click an object, tune, save: 1 vest 2 wallet 3 keys 4 bottle 5 glasses case",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(view, f"Saved: {', '.join(saved) or '-'}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in NAMES:
            save_color(NAMES[key], lower, upper)
            saved = load_colors()
            print(f"Saved {NAMES[key]}: lower={lower} upper={upper}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()