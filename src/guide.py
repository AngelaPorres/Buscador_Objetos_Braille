"""Hot/cold guide: decides which hint to give from the distance between the person and the object."""
from collections import deque
 
import numpy as np
 
# Distances in pixels of the undistorted image (tune them in the room with test_guide.py)
FOUND_DISTANCE = 80     # closer than this = the person is at the object
FOUND_FRAMES = 20       # ... for this many frames in a row (about 0.7 s at 30 FPS)
BURNING_DISTANCE = 160  # closer than this = "burning"
MIN_CHANGE = 40         # smaller changes of distance do not count as getting closer or further
HINT_INTERVAL = 2.0     # seconds between two hints
LOCATOR_FRAMES = 15     # detections used to fix the position of the object
 
# What the voice says (the users of the demo speak Spanish)
OBJECT_NAMES = {"wallet": "la cartera", "keys": "las llaves",
                "bottle": "la botella", "glasses_case": "el estuche de gafas"}
PHRASES = {
    "start": "Vamos a buscar {object}. Muévete y te diré si te acercas.",
    "hotter": "Caliente",
    "colder": "Frío",
    "burning": "¡Te quemas!",
    "found": "¡Lo has encontrado!",
    "lost": "No te veo",
}
 
 
def phrase(event, object_name=None):
    """Text to say for an event."""
    return PHRASES[event].format(object=OBJECT_NAMES.get(object_name, ""))
 
 
class ObjectLocator:
    """The object stays still while it is searched: its position is the median of the last detections.
 
    This removes the jitter of the detection and keeps the position when the person hides the label.
    """
 
    def __init__(self, frames=LOCATOR_FRAMES):
        self.points = deque(maxlen=frames)
 
    def update(self, label):
        if label is not None:
            self.points.append(label["center"])
 
    @property
    def position(self):
        if not self.points:
            return None
        x, y = np.median(np.array(self.points), axis=0)
        return int(x), int(y)
 
 
class HotColdGuide:
    """Compares the distance person-object with the distance at the last hint."""
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.reference = None      # distance when the last hint was given
        self.last_hint_time = 0.0
        self.frames_close = 0
        self.person_seen = False
        self.distance = None       # current distance (shown on screen)
        self.found = False
 
    def update(self, person, target, now):
        """person and target are (x, y) or None; now is the time in seconds.
 
        Returns an event ("hotter", "colder", "burning", "found", "lost") or None.
        """
        if self.found or target is None:
            return None
 
        if person is None:
            self.distance = None
            self.frames_close = 0
            if self.person_seen:  # say it only once, when the person disappears
                self.person_seen = False
                self.reference = None
                return "lost"
            return None
        self.person_seen = True
        self.distance = float(np.hypot(person[0] - target[0], person[1] - target[1]))
 
        # Found: close to the object for several frames in a row
        self.frames_close = self.frames_close + 1 if self.distance < FOUND_DISTANCE else 0
        if self.frames_close >= FOUND_FRAMES:
            self.found = True
            return "found"
 
        # First distance (or the person has just come back): it is only the reference
        if self.reference is None:
            self.reference = self.distance
            self.last_hint_time = now
            return None
        if now - self.last_hint_time < HINT_INTERVAL:
            return None
 
        change = self.distance - self.reference
        if self.distance < BURNING_DISTANCE:
            event = "burning"
        elif change <= -MIN_CHANGE:
            event = "hotter"
        elif change >= MIN_CHANGE:
            event = "colder"
        else:
            return None  # has hardly moved: wait until there is a clear change
        self.reference = self.distance
        self.last_hint_time = now
        return event