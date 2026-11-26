"""Object detector: finds the colour square stuck on each object."""
import cv2
import numpy as np

from color_mask import hsv_mask

# Letter of the object card -> name of the object in colors.json
OBJECTS = {"C": "wallet", "L": "keys", "B": "bottle", "E": "glasses_case"}

# Shape filters (a 6 cm square is about 20x20 px when it is 3 m away)
MIN_AREA = 60          # smaller blobs are noise
MAX_AREA = 8000        # bigger blobs are not a label (e.g. a red jumper)
MAX_ELONGATION = 3.0   # a square seen at an angle looks like a rectangle, but never like a stripe
SHAPE_AREA = 600       # from this area on (about 25x25 px) the blob must also have 4 corners

KERNEL = np.ones((3, 3), np.uint8)


def clean_mask(mask):
    """Opening removes isolated pixels and closing fills small holes (erosion + dilation)."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)


def looks_like_label(contour, area):
    """Shape test: not elongated and, if it is big enough to tell, with 4 corners."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) == 0 or max(w, h) / min(w, h) > MAX_ELONGATION:
        return False
    if area < SHAPE_AREA:
        return True  # too small to count corners reliably: colour and size decide
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    return len(approx) == 4 and cv2.isContourConvex(approx)


def find_label(hsv, color_range):
    """Find the colour label of one object in an HSV frame.

    Returns (mask, label). label is None if nothing is found, otherwise a dict with
    "center" (x, y), "box" (x, y, w, h) and "area".
    """
    mask = clean_mask(hsv_mask(hsv, color_range["lower"], color_range["upper"]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_AREA <= area <= MAX_AREA or not looks_like_label(contour, area):
            continue
        if best is None or area > best["area"]:  # keep the biggest valid blob
            x, y, w, h = cv2.boundingRect(contour)
            best = {"center": (x + w // 2, y + h // 2), "box": (x, y, w, h), "area": area}
    return mask, best