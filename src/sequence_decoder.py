"""Password decoder: remembers the braille letters shown one after another."""
import time

PASSWORD = "GUIA"
FRAMES_TO_CONFIRM = 8   # a letter counts only if it is seen in this many consecutive frames
TIMEOUT = 15.0          # seconds allowed between two letters before the sequence is forgotten


class LetterStabilizer:
    """Confirms a letter only when it is detected in several consecutive frames."""

    def __init__(self, frames_to_confirm=FRAMES_TO_CONFIRM):
        self.frames_to_confirm = frames_to_confirm
        self.candidate = None
        self.count = 0

    def update(self, letter):
        """Return the letter on the frame it becomes confirmed, otherwise None."""
        if letter is None or letter != self.candidate:
            self.candidate = letter
            self.count = 0 if letter is None else 1
            return None
        self.count += 1
        return letter if self.count == self.frames_to_confirm else None


class SequenceDecoder:
    """Stores up to len(password) letters and grants access if they arrive in the right order."""

    def __init__(self, password=PASSWORD, timeout=TIMEOUT):
        self.password = password
        self.timeout = timeout
        self.stabilizer = LetterStabilizer()
        self.reset()

    def reset(self):
        """Forget everything entered so far."""
        self.sequence = []          # letters accepted so far
        self.last_confirmed = None  # last confirmed letter (right or wrong)
        self.last_time = None       # time of the last accepted letter
        self.granted = False
        self.frames_without_card = 0

    def time_left(self, now=None):
        """Seconds left to show the next letter (only meaningful while a sequence is in progress)."""
        if not self.sequence:
            return self.timeout
        now = time.monotonic() if now is None else now
        return max(0.0, self.timeout - (now - self.last_time))

    def update(self, letter, now=None):
        """Feed the letter detected in the current frame (or None).

        Returns an event: "accepted", "wrong", "timeout", "granted", or None if nothing happened.
        """
        if self.granted:
            return None
        now = time.monotonic() if now is None else now

        # Forget the sequence if the person takes too long to show the next card
        if self.sequence and now - self.last_time > self.timeout:
            self.reset()
            return "timeout"

        # Allow a new card after the previous one has been removed
        if letter is None:
            self.frames_without_card += 1

            if self.frames_without_card >= 6:
                self.last_confirmed = None
        else:
            self.frames_without_card = 0

        confirmed = self.stabilizer.update(letter)
        if confirmed is None or confirmed == self.last_confirmed:
            return None  # nothing new, or still the same card as before
        self.last_confirmed = confirmed

        if confirmed == self.password[len(self.sequence)]:
            self.sequence.append(confirmed)
            self.last_time = now
            if len(self.sequence) == len(self.password):
                self.granted = True
                return "granted"
            return "accepted"

        # Wrong letter: forget the sequence (last_confirmed is kept so the same
        # wrong card is not reported again while it is still being shown)
        self.sequence = []
        self.last_time = None
        return "wrong"