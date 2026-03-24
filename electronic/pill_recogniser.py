"""
Pill detection and counting module using Claude Vision API.

Provides the PillRecogniser class for using Claude's vision capabilities
to count pills, verify dispenses, and check tray emptiness.

Frames are always passed in from outside — this module never opens a camera.
"""
import anthropic
import base64
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.hardware_config import ANTHROPIC_API_KEY, CLAUDE_MODEL

import numpy as np


class PillRecogniser:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model  = CLAUDE_MODEL

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert CV2 frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.standard_b64encode(buffer).decode('utf-8')

    def count_pills(self, frame: np.ndarray, debug: bool = False) -> tuple:
        """
        Use Claude vision to count pills in frame.
        Returns (count, description).
        Frame must be provided — this method never opens a camera.
        """
        if frame is None:
            return 0, "No frame provided"

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

        try:
            lines       = response.strip().split('\n')
            count       = int(lines[0].replace('COUNT:', '').strip())
            description = lines[1].replace('DESCRIPTION:', '').strip()
        except Exception as e:
            print(f"Parse error: {e}")
            count       = 0
            description = response

        return count, description

    def is_tray_empty(self, frame: np.ndarray, debug: bool = False) -> bool:
        """Check tray is empty before dispensing."""
        count, description = self.count_pills(frame=frame, debug=debug)
        print(f"Tray empty check: {count} pills — {description}")
        return count == 0

    def verify_dispense(
        self, frame: np.ndarray, expected_count: int = 1, debug: bool = False
    ) -> tuple:
        """
        Verify correct number of pills dispensed.
        Returns (success, detected_count, description).
        """
        count, description = self.count_pills(frame=frame, debug=debug)
        success = count == expected_count
        print(f"Dispense verify: expected={expected_count}, detected={count}, success={success}")
        return success, count, description