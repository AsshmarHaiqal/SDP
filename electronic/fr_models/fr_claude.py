"""
Model 2: Claude Vision API facial recognition.
Sends captured frame to Claude for identity verification.
Requires internet. Fallback to local if offline.
"""
import base64
import json
import os
import socket
import time

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    print("WARNING: anthropic not installed")

import cv2
import numpy as np

from config.hardware_config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .base import FaceRecognitionBase



class ClaudeFaceRecognition(FaceRecognitionBase):
    """
    Stores reference face images (not encodings).
    Sends both reference + live capture to Claude for comparison.
    """

    def __init__(
        self,
        faces_dir: str,
        api_key: str,
        camera_index: int = 0,
        model: str = CLAUDE_MODEL
    ):
        self.faces_dir    = faces_dir
        self.camera_index = camera_index
        self.model        = model
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if _ANTHROPIC_AVAILABLE else None
        os.makedirs(faces_dir, exist_ok=True)

    # ── Enrollment ────────────────────────────────────────────────────

    def enroll(self, name: str, image_path: str = None) -> bool:
        """Save reference image for Claude to compare against."""
        if image_path:
            frame = cv2.imread(image_path)
        else:
            frame = self._capture_frame_bgr()

        if frame is None:
            print("Could not capture enrollment image")
            return False

        save_path = os.path.join(self.faces_dir, f"{name}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Enrolled (Claude mode): {name} → {save_path}")
        return True

    # ── Verification ──────────────────────────────────────────────────

    def verify(self, name: str, override: bool = False) -> bool:
        if override:
            return True
        if not _ANTHROPIC_AVAILABLE:
            print("Anthropic not available")
            return False
        if not self._has_internet():
            print("No internet — Claude verification unavailable")
            return False

        reference_path = os.path.join(self.faces_dir, f"{name}.jpg")
        if not os.path.exists(reference_path):
            print(f"No enrollment found for: {name}")
            return False

        live_frame = self._capture_frame_bgr()
        if live_frame is None:
            return False

        return self._verify_with_claude(live_frame, reference_path, name)

    def verify_live(self, name: str, override: bool = False) -> bool:
        """
        Shows live feed, captures on keypress, sends to Claude.
        """
        if override:
            return True

        reference_path = os.path.join(self.faces_dir, f"{name}.jpg")
        if not os.path.exists(reference_path):
            print(f"No enrollment found for: {name}")
            return False

        cap = cv2.VideoCapture(self.camera_index)
        print("Press SPACE to capture and verify, Q to quit")
        captured_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, "SPACE to verify | Q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("PillWheel - Claude Verification", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                captured_frame = frame.copy()
                cv2.putText(display, "Verifying with Claude...",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow("PillWheel - Claude Verification", display)
                cv2.waitKey(500)
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            return False

        print("Sending to Claude for verification...")
        result = self._verify_with_claude(captured_frame, reference_path, name)
        print(f"Claude result: {'Verified' if result else 'Not verified'}")
        return result

    def list_enrolled(self) -> list:
        if not os.path.exists(self.faces_dir):
            return []
        return [f.replace(".jpg", "") for f in os.listdir(self.faces_dir)
                if f.endswith(".jpg")]

    # ── Internal ──────────────────────────────────────────────────────

    def _verify_with_claude(self, live_frame, reference_path: str, name: str) -> bool:
        try:
            live_b64 = self._encode_frame(live_frame)
            ref_b64  = self._encode_file(reference_path)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Image 1 is a reference photo of an enrolled person."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": ref_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Image 2 is a live camera capture."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": live_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Are these the same person?
Consider: same facial structure, eyes, nose, jawline.
Allow for different lighting, angle, expression.

Respond ONLY with this JSON, nothing else:
{
    "match": true or false,
    "confidence": "high" | "medium" | "low",
    "reason": "<one sentence>"
}"""
                        }
                    ]
                }]
            )

            raw    = response.content[0].text.strip()
            parsed = json.loads(raw)

            print(f"  Claude: match={parsed['match']} "
                  f"confidence={parsed['confidence']} "
                  f"reason={parsed['reason']}")

            # Reject low-confidence matches for safety
            if parsed["confidence"] == "low":
                print("  Confidence too low — rejecting")
                return False

            return bool(parsed["match"])

        except json.JSONDecodeError as e:
            print(f"Claude returned invalid JSON: {e}")
            return False
        except anthropic.APIError as e:
            print(f"Claude API error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def _encode_frame(self, frame_bgr) -> str:
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _encode_file(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _capture_frame_bgr(self):
        cap = cv2.VideoCapture(self.camera_index)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _has_internet(self) -> bool:
        try:
            socket.setdefaulttimeout(2)
            socket.connect(("8.8.8.8", 53))
            return True
        except OSError:
            return False