"""Text to speech with pyttsx3 in a background thread, so the video does not freeze while talking."""
import queue
import threading
 
import pyttsx3
 
VOICE_HINT = "ES-ES"   # use the first installed voice whose id contains this text (Spanish from Spain)
SPEECH_RATE = 170      # words per minute
 
 
class Speaker:
    """Says one message at a time. If a new message arrives while talking, only the newest one waits."""
 
    def __init__(self, volume=1.0):
        self.volume = volume
        self.messages = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
 
    def say(self, text):
        """Queue a message and return at once (the old waiting message, if any, is dropped)."""
        self._replace(text)
 
    def close(self):
        """Stop the voice thread (it finishes the sentence it is saying)."""
        self._replace(None)
        self.thread.join(timeout=5)
 
    def _replace(self, item):
        try:
            self.messages.get_nowait()  # drop the message that is still waiting
        except queue.Empty:
            pass
        self.messages.put_nowait(item)
 
    def _run(self):
        # The engine is created inside the thread that uses it
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            if VOICE_HINT in voice.id.upper():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", SPEECH_RATE)
        engine.setProperty("volume", self.volume)
        while True:
            text = self.messages.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()