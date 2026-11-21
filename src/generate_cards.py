"""Generate the braille card images, ready to print (one per A4 sheet)."""
from pathlib import Path

import cv2
import numpy as np

from braille import (BOTTOM_BAND, CARD_HEIGHT, CARD_WIDTH, DOT_SIZE, FRAME,
                     MARGIN, letter_to_pattern)

# Password (GUIA) + objects (C wallet, L keys, B bottle, P pill box)
LETTERS = ["G", "U", "I", "A", "C", "L", "B", "P"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cards"
WHITE, BLACK = 255, 0


def draw_card(letter):
    """Draw the card for one letter and return it as a grayscale image."""
    img = np.full((CARD_HEIGHT, CARD_WIDTH), WHITE, dtype=np.uint8)

    # Outer black rectangle (this becomes the frame)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = CARD_WIDTH - MARGIN, CARD_HEIGHT - MARGIN
    cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, thickness=-1)

    # White interior: the frame stays thicker at the bottom (orientation band)
    ix0, iy0 = x0 + FRAME, y0 + FRAME
    ix1, iy1 = x1 - FRAME, y1 - BOTTOM_BAND
    cv2.rectangle(img, (ix0, iy0), (ix1, iy1), WHITE, thickness=-1)

    # 2 columns x 3 rows grid: one black square for each active dot
    cell_w = (ix1 - ix0) / 2
    cell_h = (iy1 - iy0) / 3
    for i, active in enumerate(letter_to_pattern(letter)):  # i = 0..5 -> dots 1..6
        if not active:
            continue
        col, row = divmod(i, 3)  # dots 1-3 -> column 0, dots 4-6 -> column 1
        cx = ix0 + (col + 0.5) * cell_w
        cy = iy0 + (row + 0.5) * cell_h
        top_left = (int(cx - DOT_SIZE / 2), int(cy - DOT_SIZE / 2))
        bottom_right = (int(cx + DOT_SIZE / 2), int(cy + DOT_SIZE / 2))
        cv2.rectangle(img, top_left, bottom_right, BLACK, thickness=-1)

    # White letter inside the bottom band (only for sighted people; the detector ignores it)
    cv2.putText(img, letter, (ix0, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 3, WHITE, 8)
    return img


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for letter in LETTERS:
        path = OUTPUT_DIR / f"card_{letter}.png"
        cv2.imwrite(str(path), draw_card(letter))
        print("Saved:", path)


if __name__ == "__main__":
    main()