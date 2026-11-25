"""HSV colour ranges: masks with cv2.inRange and loading/saving the tuned ranges."""
import json
from pathlib import Path

import cv2
import numpy as np

# Tuned ranges are stored in the project root
COLORS_PATH = Path(__file__).resolve().parent.parent / "colors.json"


def hsv_mask(hsv, lower, upper):
    """Binary mask of the pixels inside an HSV range.

    If the lower hue is greater than the upper hue, the range wraps around
    the end of the hue circle (needed for red, which is near both 0 and 179).
    """
    lower, upper = np.array(lower), np.array(upper)
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    # Wrap-around: [lower hue, 179] OR [0, upper hue]
    mask_high = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]]))
    mask_low = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]), upper)
    return cv2.bitwise_or(mask_high, mask_low)


def load_colors(path=COLORS_PATH):
    """Return the saved ranges as {name: {"lower": [h, s, v], "upper": [h, s, v]}}."""
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_color(name, lower, upper, path=COLORS_PATH):
    """Save (or overwrite) the range of one colour."""
    colors = load_colors(path)
    colors[name] = {"lower": [int(v) for v in lower], "upper": [int(v) for v in upper]}
    Path(path).write_text(json.dumps(colors, indent=2), encoding="utf-8")