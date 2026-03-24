"""
electronic/sound_actuator.py
Text-to-speech audio feedback for PillWheel stage transitions.

Models:
    1 = Piper TTS    (local, neural, natural — recommended)
    2 = Edge TTS     (online, Microsoft neural voices, very natural)
    3 = eSpeak       (local, robotic — fallback only)

Setup:
    See install instructions above.

    # Piper (local neural TTS)
pip install piper-tts
# Download a voice model (British English — suits care home context)
mkdir -p voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx -P voices/
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json -P voices/

# Edge TTS (online, Microsoft neural voices)
pip install edge-tts

# Keep espeak as last resort fallback
sudo apt install espeak-ng

Usage:
    from electronic.sound_actuator import SoundActuator
    sound = SoundActuator()
    sound.verifying_face()
    sound.speak("Custom message here")
"""

import os
import subprocess
import tempfile
import threading
import asyncio

# ════════════════════════════════════════════════════════════════════
#  SWITCH TTS MODEL HERE
#  1 = Piper (local neural)     — best for Pi, no internet needed
#  2 = Edge TTS (online neural) — best quality, needs internet
#  3 = eSpeak (local fallback)  — robotic, always available
# ════════════════════════════════════════════════════════════════════
TTS_MODEL = 1

# ── Piper config ──────────────────────────────────────────────────────────────
PIPER_VOICE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "voices", "en_GB-alba-medium.onnx"
)

# ── Edge TTS config ───────────────────────────────────────────────────────────
# Full list: run `edge-tts --list-voices` and look for en-GB voices
EDGE_VOICE = "en-GB-SoniaNeural"     # Natural British female
# EDGE_VOICE = "en-GB-RyanNeural"    # Natural British male

# ── eSpeak config ─────────────────────────────────────────────────────────────
ESPEAK_RATE   = 140    # words per minute — slightly slower = clearer for elderly
ESPEAK_VOICE  = "en-gb"

# ── Phrases ───────────────────────────────────────────────────────────────────
# Written conversationally — short sentences, natural pauses via punctuation.
# Avoid abbreviations and acronyms; TTS reads them letter by letter.

PHRASES = {
    "ready":          "Your medication is ready.",
    "verifying":      "Please look at the camera. Verifying your identity.",
    "verified":       "Identity confirmed. Welcome.",
    "access_denied":  "Sorry, I could not recognise your face. Please try again, or ask a carer for help.",
    "dispensing":     "Dispensing your medication now. Please wait a moment.",
    "take_with_food": "Remember to take your medication with food or water.",
    "collected":      "Your medication has been dispensed. Have a lovely day.",
    "error":          "Something has gone wrong. Please press the call button or ask a carer for assistance.",
    "missed_dose":    "A dose was missed. A carer has been notified.",
    "low_stock":      "Medication stock is running low. A carer has been notified to refill the dispenser.",
    "no_prescription": "No prescription was found for your account. Please speak to a carer.",
    "count_mismatch": "There was an issue with your medication. A carer has been notified.",
}


# ════════════════════════════════════════════════════════════════════
#  TTS BACKENDS
# ════════════════════════════════════════════════════════════════════

class _PiperTTS:
    """
    Local neural TTS using Piper.
    Sounds natural, runs fully offline on Pi 3.
    ~1s latency on first call, faster after warmup.
    """

    def __init__(self):
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(PIPER_VOICE)
            self._available = True
            print(f"Piper TTS loaded: {os.path.basename(PIPER_VOICE)}")
        except Exception as e:
            print(f"Piper TTS unavailable: {e}")
            self._available = False

    def speak(self, text: str):
        if not self._available:
            return False
        tmp_path = None
        try:
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            with wave.open(tmp_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            subprocess.run(
                ["aplay", "-q", "-D", "plughw:2,0", tmp_path],
                check=True,
                timeout=30
            )
            os.unlink(tmp_path)
            return True

        except Exception as e:
            print(f"Piper speak error: {e}")
            try:
                if tmp_path:
                    os.unlink(tmp_path)
            except Exception:
                pass
            return False
    
    



class _EdgeTTS:
    """
    Online neural TTS using Microsoft Edge voices.
    Sounds very natural — closest to a real human voice.
    Requires internet connection.
    """

    def __init__(self):
        try:
            import edge_tts
            self._edge_tts = edge_tts
            self._available = True
            print(f"Edge TTS ready: {EDGE_VOICE}")
        except ImportError:
            print("Edge TTS unavailable — pip install edge-tts")
            self._available = False

    def speak(self, text: str) -> bool:
        if not self._available:
            return False
        if not self._has_internet():
            print("Edge TTS: no internet")
            return False
        try:
            asyncio.run(self._speak_async(text))
            return True
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return False

    async def _speak_async(self, text: str):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            communicate = self._edge_tts.Communicate(text, EDGE_VOICE)
            await communicate.save(tmp_path)
            subprocess.run(
                ["mpg123", "-q", tmp_path],
                check=True,
                timeout=30
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _has_internet(self) -> bool:
        import socket
        try:
            socket.setdefaulttimeout(2)
            socket.connect(("8.8.8.8", 53))
            return True
        except OSError:
            return False


class _ESpeakTTS:
    """
    Local fallback using eSpeak-NG.
    Robotic but always available — never fails.
    """

    def __init__(self):
        self._available = True
        print("eSpeak TTS ready (fallback mode)")

    def speak(self, text: str) -> bool:
        try:
            subprocess.run(
                [
                    "espeak-ng",
                    "-v", ESPEAK_VOICE,
                    "-s", str(ESPEAK_RATE),
                    text
                ],
                check=True,
                timeout=30
            )
            return True
        except Exception as e:
            print(f"eSpeak error: {e}")
            return False


# ════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ════════════════════════════════════════════════════════════════════

class SoundActuator:
    """
    Plays spoken audio at each stage of the dispense workflow.
    All calls are non-blocking — audio plays in a background thread.

    Switch voice engine via TTS_MODEL constant at top of file:
        1 = Piper (local neural)
        2 = Edge TTS (online)
        3 = eSpeak (fallback)
    """

    def __init__(self):
        self._engine  = self._load_engine()
        self._lock    = threading.Lock()  # Prevent overlapping speech

    def _load_engine(self):
        if TTS_MODEL == 1:
            engine = _PiperTTS()
            # Fall back to eSpeak if Piper unavailable
            if not engine._available:
                print("Falling back to eSpeak")
                return _ESpeakTTS()
            return engine

        elif TTS_MODEL == 2:
            engine = _EdgeTTS()
            if not engine._available:
                print("Falling back to eSpeak")
                return _ESpeakTTS()
            return engine

        elif TTS_MODEL == 3:
            return _ESpeakTTS()

        else:
            raise ValueError(f"Invalid TTS_MODEL: {TTS_MODEL}")

    # ── Stage methods ─────────────────────────────────────────────────

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

    def missed_dose(self):
        self.speak(PHRASES["missed_dose"])

    def low_stock(self):
        self.speak(PHRASES["low_stock"])

    def no_prescription(self):
        self.speak(PHRASES["no_prescription"])

    def count_mismatch(self):
        self.speak(PHRASES["count_mismatch"])

    # ── Core ──────────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak any text in a background thread.
        If audio is already playing, waits for it to finish first.
        """
        threading.Thread(
            target=self._speak_blocking,
            args=(text,),
            daemon=True
        ).start()

    def speak_wait(self, text: str):
        """
        Blocking version — waits for speech to finish before returning.
        Use when timing matters (e.g. before tray tilts).
        """
        self._speak_blocking(text)

    def _speak_blocking(self, text: str):
        with self._lock:  # Queue speech — never overlap
            self._engine.speak(text)