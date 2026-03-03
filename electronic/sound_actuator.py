"""
electronic/sound_actuator.py
Text-to-speech audio feedback for PillWheel stage transitions.

Setup on Raspberry Pi:
    sudo apt install espeak
    pip install pyttsx3

USB speaker:
    Plug in, then set as default output:
    sudo raspi-config → System → Audio → USB Audio

Usage:
    from electronic.sound_actuator import SoundActuator
    sound = SoundActuator()
    sound.verifying_face()
    sound.dispensing()
"""

import threading

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


# ── Phrases ───────────────────────────────────────────────────────────────────

PHRASES = {
    "ready":          "Medication ready for collection",
    "verifying":      "Verifying face, please look at the camera",
    "verified":       "Identity verified",
    "access_denied":  "Face not recognised, please try again",
    "dispensing":     "Dispensing medication, please wait",
    "take_with_food": "Please take your medication with food",
    "collected":      "Collection complete, have a great day",
    "error":          "An error has occurred, please call for assistance",
}


# ── Actuator ──────────────────────────────────────────────────────────────────

class SoundActuator:
    """
    Plays spoken audio cues at each stage of the dispense workflow.
    All calls are non-blocking — audio plays in a background thread
    so the UI never freezes waiting for speech to finish.
    """

    def __init__(self, rate: int = 120, volume: float = 1.0):
        self.rate   = rate    # words per minute (default espeak: 200)
        self.volume = volume  # 0.0 – 1.0

        if not _TTS_AVAILABLE:
            print("⚠️  pyttsx3 not installed – audio cues disabled")

    # ── Stage methods ─────────────────────────────────────────────────────────

    def ready_for_collection(self):
        self.speak(PHRASES["ready"])

    def verifying_face(self):
        self.speak(PHRASES["verifying"])

    def verified(self):
        self.speak(PHRASES["verified"])

    def access_denied(self):
        self.speak(PHRASES["access_denied"])

    def dispensing(self):
        self.speak(PHRASES["dispensing"])

    def take_with_food(self):
        self.speak(PHRASES["take_with_food"])

    def collected(self):
        self.speak(PHRASES["collected"])

    def error(self):
        self.speak(PHRASES["error"])

    # ── Core ──────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Speak any arbitrary text in a background thread."""
        if not _TTS_AVAILABLE:
            return
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    import subprocess

    def _speak(self, text: str):
        subprocess.run(["espeak-ng", text])
