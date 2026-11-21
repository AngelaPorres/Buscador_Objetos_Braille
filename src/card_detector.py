"""Braille card detector: finds a card in a frame and decodes its letter."""
import cv2
import numpy as np

from braille import (BOTTOM_BAND, CARD_HEIGHT, CARD_WIDTH, DOT_SIZE, FRAME,
                     MARGIN, PATTERN_TO_LETTER)

# Size of the black frame (outer edge) in card design units
FRAME_W = CARD_WIDTH - 2 * MARGIN    # 1214
FRAME_H = CARD_HEIGHT - 2 * MARGIN   # 1800

# The detected card is warped to a front view of this size (design units scaled down)
WARP_SCALE = 0.25
WARP_W = int(FRAME_W * WARP_SCALE)
WARP_H = int(FRAME_H * WARP_SCALE)

# Detection parameters (tune them if needed)
CANNY_LOW, CANNY_HIGH = 50, 150       # Canny hysteresis thresholds
MIN_AREA = 3000                       # minimum contour area in pixels (filters out noise)
ASPECT_MIN, ASPECT_MAX = 0.45, 0.95   # accepted width/height ratio of the card frame
MIN_CONTRAST = 40                     # minimum grey-level difference between black and white parts
FRAME_THRESHOLD = 0.7                 # fraction of black pixels required on the frame borders
DOT_THRESHOLD = 0.5                   # fraction of black pixels to count a cell as a dot


def _dot_rois():
    """Regions (x0, y0, x1, y1) of the warped card where the 6 dots are checked."""
    cell_w = (FRAME_W - 2 * FRAME) / 2
    cell_h = (FRAME_H - FRAME - BOTTOM_BAND) / 3
    half = DOT_SIZE * 0.25 * WARP_SCALE  # we only look at the centre of each dot
    rois = []
    for i in range(6):  # i = 0..5 -> dots 1..6
        col, row = divmod(i, 3)  # dots 1-3 -> column 0, dots 4-6 -> column 1
        cx = (FRAME + (col + 0.5) * cell_w) * WARP_SCALE
        cy = (FRAME + (row + 0.5) * cell_h) * WARP_SCALE
        rois.append((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    return rois


DOT_ROIS = _dot_rois()


def _dark_fraction(region):
    """Fraction of black pixels (value 255 in an inverted binary image)."""
    if region.size == 0:
        return 0.0
    return np.count_nonzero(region) / region.size


def _order_corners(quad):
    """Order 4 points clockwise starting at the top-left corner."""
    pts = quad.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]           # clockwise (the image y axis points down)
    start = np.argmin(pts.sum(axis=1))      # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)


def _quads(edges):
    """Return (area, contour) for every closed contour with 4 corners in an edge image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((area, approx))
    return quads


def _find_candidates(gray):
    """Find the edges of the image and return the 4-corner contours, largest first."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    # Far cards: their thin frame is only well separated in the raw edges.
    # Near cards: dilation joins small gaps so the outline becomes a closed contour.
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    candidates = _quads(edges) + _quads(dilated)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return blur, edges, [approx for _, approx in candidates]


def _warp_card(gray, quad):
    """Warp a quadrilateral to a front (portrait) view. Returns (warped, corners) or (None, None)."""
    corners = _order_corners(quad)
    tl, tr, br, bl = corners
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width > height:
        # Card seen in landscape: shift the corner order so the long side becomes vertical
        corners = np.roll(corners, 1, axis=0)
        width, height = height, width
    if not ASPECT_MIN <= width / height <= ASPECT_MAX:
        return None, None

    target = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
                      dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(gray, matrix, (WARP_W, WARP_H))
    return warped, corners


def _read_card(warped):
    """Check that it is one of our cards, fix the orientation and read the 6 dots.

    Returns (pattern, card_binary). pattern is None if it is not a valid card.
    """
    # Otsu works well here: the warped card only has black and white
    _, card_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 0) There must be a clear difference between the black and the white parts
    dark, light = warped[card_bin > 0], warped[card_bin == 0]
    if dark.size == 0 or light.size == 0 or light.mean() - dark.mean() < MIN_CONTRAST:
        return None, card_bin

    # 1) The four outer borders must be black (the frame)
    b = max(int(FRAME * WARP_SCALE * 0.6), 1)
    borders = [card_bin[:b, :], card_bin[-b:, :], card_bin[:, :b], card_bin[:, -b:]]
    if min(_dark_fraction(border) for border in borders) < FRAME_THRESHOLD:
        return None, card_bin

    # 2) Orientation: the thick band must be at the bottom. Just inside the top frame
    #    the card is white, while at the same distance from the bottom it is still black.
    y0, y1 = int(FRAME * 1.15 * WARP_SCALE), int(FRAME * 2.2 * WARP_SCALE)
    x0, x1 = int(FRAME * 1.2 * WARP_SCALE), WARP_W - int(FRAME * 1.2 * WARP_SCALE)
    top_dark = _dark_fraction(card_bin[y0:y1, x0:x1])
    bottom_dark = _dark_fraction(card_bin[WARP_H - y1:WARP_H - y0, x0:x1])
    if top_dark > bottom_dark:
        card_bin = cv2.rotate(card_bin, cv2.ROTATE_180)  # card was upside down
        top_dark, bottom_dark = bottom_dark, top_dark
    if bottom_dark < FRAME_THRESHOLD or top_dark > 1 - FRAME_THRESHOLD:
        return None, card_bin  # no thick band at the bottom: not one of our cards

    # 3) Read the 2x3 grid by slicing the centre of each cell
    pattern = tuple(_dark_fraction(card_bin[ry0:ry1, rx0:rx1]) > DOT_THRESHOLD
                    for rx0, ry0, rx1, ry1 in DOT_ROIS)
    return pattern, card_bin


def detect_card(frame, debug=False):
    """Look for a braille card in a BGR frame.

    Returns (letter, corners, steps):
      - letter: decoded letter, or None if there is no card or the pattern is unknown
      - corners: 4x2 array with the card corners in the frame, or None if no card was found
      - steps: intermediate images of the process (only filled when debug=True)
    """
    steps = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur, edges, candidates = _find_candidates(gray)

    if debug:
        steps["01_gray"] = gray
        steps["02_blur"] = blur
        steps["03_edges"] = edges
        candidates_img = frame.copy()
        cv2.drawContours(candidates_img, candidates, -1, (0, 0, 255), 2)
        steps["04_candidates"] = candidates_img

    best = None  # first valid card found, in case none of them gives a known letter
    for quad in candidates:
        warped, corners = _warp_card(gray, quad)
        if warped is None:
            continue
        pattern, card_bin = _read_card(warped)
        if pattern is None:
            continue  # it has 4 corners but it is not one of our cards
        letter = PATTERN_TO_LETTER.get(pattern)
        if best is None or letter is not None:
            best = (letter, corners, warped, pattern, card_bin)
        if letter is not None:
            break

    if best is None:
        return None, None, steps

    letter, corners, warped, pattern, card_bin = best
    if debug:
        steps["05_warped"] = warped
        card_vis = cv2.cvtColor(card_bin, cv2.COLOR_GRAY2BGR)
        for active, (rx0, ry0, rx1, ry1) in zip(pattern, DOT_ROIS):
            color = (0, 255, 0) if active else (0, 0, 255)
            cv2.rectangle(card_vis, (rx0, ry0), (rx1, ry1), color, 2)
        steps["06_card_binary"] = card_vis
    return letter, corners, steps