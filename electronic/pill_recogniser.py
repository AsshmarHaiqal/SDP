"""
Pill detection and counting module using Claude Vision API and a USB webcam.

Provides the PillRecogniser class for capturing webcam frames and using
Claude's vision capabilities to count pills, verify dispenses, and check
tray emptiness on Raspberry Pi hardware.
"""
import anthropic
import base64
import cv2
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.hardware_config import ANTHROPIC_API_KEY, CLAUDE_MODEL


class PillRecogniser:
    def __init__(self, camera_index='/dev/video0'):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.camera_index = camera_index
        self.model = CLAUDE_MODEL

    def capture_frame(self):
        """Capture a single frame from webcam."""
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        time.sleep(1)  # warmup
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not capture frame")
            return None
        return frame

    def frame_to_base64(self, frame):
        """Convert CV2 frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.standard_b64encode(buffer).decode('utf-8')

    def count_pills(self, frame=None, debug=False):
        """
        Use Claude vision to count pills in frame.
        Returns (count, description)
        """
        if frame is None:
            frame = self.capture_frame()
        if frame is None:
            return 0, "Camera error"

        image_data = self.frame_to_base64(frame)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """You are a pill counting assistant for a medical dispenser.
Count the number of pills/tablets/capsules visible in this image.
Respond in this exact format:
COUNT: <number>
DESCRIPTION: <brief description of what you see>

If no pills are visible, respond:
COUNT: 0
DESCRIPTION: No pills detected"""
                        }
                    ],
                }
            ],
        )

        response = message.content[0].text
        if debug:
            print(f"Claude response: {response}")

        # Parse response
        try:
            lines = response.strip().split('\n')
            count = int(lines[0].replace('COUNT:', '').strip())
            description = lines[1].replace('DESCRIPTION:', '').strip()
        except Exception as e:
            print(f"Parse error: {e}")
            count = 0
            description = response

        return count, description

    def is_tray_empty(self, debug=False):
        """Check tray is empty before dispensing."""
        count, description = self.count_pills(debug=debug)
        print(f"Tray empty check: {count} pills - {description}")
        return count == 0

    def verify_dispense(self, expected_count=1, debug=False):
        """
        Verify correct number of pills dispensed.
        Returns (success, detected_count, description)
        """
        count, description = self.count_pills(debug=debug)
        success = count == expected_count
        print(f"Dispense verify: expected={expected_count}, detected={count}, success={success}")
        return success, count, description


# --- Test ---
if __name__ == "__main__":
    recogniser = PillRecogniser()

    print("=== Pill Recognition Test ===\n")

    print("1. Testing empty tray...")
    empty = recogniser.is_tray_empty(debug=True)
    print(f"   Tray empty: {empty}\n")

    input("2. Place 1 pill in tray then press Enter...")
    success, count, desc = recogniser.verify_dispense(expected_count=1, debug=True)
    print(f"   Success: {success}, Count: {count}, Description: {desc}\n")

    input("3. Place 2 pills in tray then press Enter...")
    success, count, desc = recogniser.verify_dispense(expected_count=2, debug=True)
    print(f"   Success: {success}, Count: {count}, Description: {desc}\n")

    print("Test complete!")
