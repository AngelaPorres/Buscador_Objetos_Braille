"""Braille alphabet and card geometry (shared by the card generator and the detector)."""

# Active dots for each letter.
# Braille numbering: dots 1-2-3 in the left column (top to bottom), 4-5-6 in the right column
BRAILLE_DOTS = {
    "A": "1", "B": "12", "C": "14", "D": "145", "E": "15",
    "F": "124", "G": "1245", "H": "125", "I": "24", "J": "245",
    "K": "13", "L": "123", "M": "134", "N": "1345", "O": "135",
    "P": "1234", "Q": "12345", "R": "1235", "S": "234", "T": "2345",
    "U": "136", "V": "1236", "W": "2456", "X": "1346", "Y": "13456",
    "Z": "1356", "Ñ": "12456",
}


def letter_to_pattern(letter):
    """Convert a letter into a 6-value tuple (True if dot 1..6 is active)."""
    dots = BRAILLE_DOTS[letter]
    return tuple(str(n) in dots for n in range(1, 7))


# Reverse dictionary: 6-value pattern -> letter (used by the detector)
PATTERN_TO_LETTER = {letter_to_pattern(letter): letter for letter in BRAILLE_DOTS}


# ---- Card geometry (in pixels, A4 aspect ratio) ----
CARD_WIDTH, CARD_HEIGHT = 1414, 2000
MARGIN = 100        # outer white border: lets you hold the card without covering the frame
FRAME = 80          # black frame thickness (top, left and right)
BOTTOM_BAND = 240   # thicker bottom band: tells which side is "down"
DOT_SIZE = 260      # side length of each black square (braille dot)