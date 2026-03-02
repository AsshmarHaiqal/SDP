"""
test_pill_detection.py
Laptop test harness for PillDetector.

Modes
-----
  python test_pill_detection.py preview   — live camera feed with detection overlay
  python test_pill_detection.py flow      — full simulated dispense session
  python test_pill_detection.py count     — single-shot: capture, count, save image

Run from the SDP/ directory:
  cd /Users/asshmarhaiqal/PillWheel/SDP
  python test_pill_detection.py preview

Tips
----
- Use coins, buttons, or M&M sweets as pill stand-ins.
- Make sure the object diameter on-screen is between 5–50 px
  (move closer/further until circles are detected).
- If nothing detects, run 'preview' and watch the console output — it
  prints the param2 threshold so you can judge whether to lower it.
"""

import sys
import time
import cv2
import numpy as np
import electronic.pill_detection as _pd_module

# ── Laptop overrides ──────────────────────────────────────────────────────────
# The Pi uses camera index 1 (USB webcam); the laptop built-in is 0.
_pd_module._CAMERA_INDEX = 0

# Speed up servo simulation delays so the flow test doesn't take 3+ mins.
import config.hardware_config as _cfg
_cfg.PILL_SETTLE_DELAY   = 1.0   # normally 3 s
_cfg.CAMERA_SETTLE_DELAY = 0.2   # normally 1 s
_cfg.TRAY_TILT_DURATION  = 1.0   # normally 3 s

# Re-import after patching so PillDetector picks up the new values.
from electronic.pill_detection import (
    PillDetector,
    TrayNotClearError,
    DispenseFailureError,
    TrayTiltFailureError,
)
from electronic.servo_controller import ServoController

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_circles(frame: np.ndarray, circles) -> np.ndarray:
    """Draw detected Hough circles onto *frame* and return a copy."""
    out = frame.copy()
    if circles is not None:
        for (x, y, r) in np.uint16(np.around(circles[0])):
            cv2.circle(out, (x, y), r, (0, 255, 0), 2)      # circle outline
            cv2.circle(out, (x, y), 2, (0, 0, 255), 3)      # centre dot
    return out


def _detect_circles(frame: np.ndarray):
    """Run the same Hough detection used by PillDetector._count_pills."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp        = _pd_module._HOUGH_DP,
        minDist   = _pd_module._HOUGH_MIN_DIST,
        param1    = _pd_module._HOUGH_PARAM1,
        param2    = _pd_module._HOUGH_PARAM2,
        minRadius = _pd_module._HOUGH_MIN_RADIUS,
        maxRadius = _pd_module._HOUGH_MAX_RADIUS,
    )
    return circles


# ── Mode: preview ─────────────────────────────────────────────────────────────

def mode_preview():
    """
    Live camera feed with pill-detection overlay.
    Use this to tune Hough parameters and check your lighting/angle.

    Controls:
      +/-   raise/lower param2 (accumulator threshold) live
      q     quit
    """
    param2 = _pd_module._HOUGH_PARAM2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera 0.")
        return

    print("\n=== PREVIEW MODE ===")
    print("Place pill-like objects (coins, buttons) in front of the camera.")
    print("Press  +  to raise detection threshold (fewer false positives)")
    print("Press  -  to lower detection threshold (catches more circles)")
    print("Press  q  to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _pd_module._HOUGH_PARAM2 = param2
        circles = _detect_circles(frame)
        count   = 0 if circles is None else len(circles[0])
        overlay = _draw_circles(frame, circles)

        label = f"Detected: {count}  |  param2={param2}  (+ / - to adjust)"
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("PillDetector — Preview (q to quit)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            param2 = min(param2 + 2, 100)
            print(f"param2 → {param2}")
        elif key == ord('-'):
            param2 = max(param2 - 2, 5)
            print(f"param2 → {param2}")

    cap.release()
    cv2.destroyAllWindows()


# ── Mode: count ───────────────────────────────────────────────────────────────

def mode_count():
    """Single-shot: capture one frame, count pills, save image, print result."""
    print("\n=== COUNT MODE ===")
    servo    = ServoController()
    detector = PillDetector(servo)

    # Manually trigger capture (bypass servo movement for quick test)
    _pd_module._CAMERA_INDEX = 0
    image, path = detector._capture_tray_image(
        resident_id    = "test_resident",
        medication_name= "test_pill",
        pill_index     = 0,
        attempt        = 1,
    )
    count = detector._count_pills(image)
    print(f"\nDetected pill count : {count}")
    print(f"Audit image saved   : {path}")

    # Show result
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=_pd_module._HOUGH_DP, minDist=_pd_module._HOUGH_MIN_DIST,
        param1=_pd_module._HOUGH_PARAM1, param2=_pd_module._HOUGH_PARAM2,
        minRadius=_pd_module._HOUGH_MIN_RADIUS,
        maxRadius=_pd_module._HOUGH_MAX_RADIUS,
    )
    annotated = _draw_circles(image, circles)
    cv2.putText(annotated, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Count result (any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    servo.cleanup()


# ── Mode: flow ────────────────────────────────────────────────────────────────

def mode_flow():
    """
    Simulated full dispense session.

    The script pauses at each step and waits for Enter so you can
    reposition objects in front of the camera to simulate:
      - Empty tray  (tray pre-check)
      - 1 pill      (after first dispense)
      - 2 pills     (after second dispense)
      - Empty tray  (after tray tilt)
    """
    print("\n=== FULL FLOW SIMULATION ===")
    print("Servo hardware not required — movements will be simulated.\n")

    servo    = ServoController()
    detector = PillDetector(servo)

    RESIDENT_ID     = "resident_001"
    PRESCRIPTION_ID = "rx_20260302_001"
    PRESCRIPTION    = [
        {"medication_name": "Vitamin_D", "dispenser_index": 0},
        {"medication_name": "Vitamin_C", "dispenser_index": 1},
    ]

    # ── Step 4: Tray pre-check ────────────────────────────────────────────────
    print("─" * 60)
    print("STEP 4 — Tray pre-check")
    print("Make sure the tray area is EMPTY, then press Enter…")
    input()
    try:
        result = detector.check_tray_empty(RESIDENT_ID)
        print(f"  ✓ Tray clear. Image: {result['image_path']}\n")
    except TrayNotClearError as e:
        print(f"  ✗ {e}")
        print("  → In production: halt, alert staff, wait for reset.")
        print("  → Returning camera to forward and exiting test.")
        detector.return_camera_to_forward()
        servo.cleanup()
        return

    # ── Step 5: Dispense + verify ─────────────────────────────────────────────
    print("─" * 60)
    print("STEP 5 — Dispense + verify")
    print(f"Prescription: {[p['medication_name'] for p in PRESCRIPTION]}")
    print()

    # For each pill the script pauses so you can add an object to the tray.
    _original_rotate = servo.rotate_dispenser

    dispense_call_count = [0]

    def _patched_rotate(idx):
        # Intercept each dispense trigger so we can prompt the tester.
        dispense_call_count[0] += 1
        expected = dispense_call_count[0]
        print(f"  [SERVO] Dispense triggered for slot {idx}.")
        print(f"  → Place {expected} pill(s) in camera view, then press Enter…")
        input()
        _original_rotate(idx)

    servo.rotate_dispenser = _patched_rotate

    try:
        summary = detector.run_dispense_session(
            RESIDENT_ID, PRESCRIPTION_ID, PRESCRIPTION
        )
        print(f"\n  ✓ All {summary['successfully_dispensed']} pill(s) verified.\n")
    except DispenseFailureError as e:
        print(f"\n  ✗ {e}")
        print("  → In production: lock machine, alert staff.")
        detector.return_camera_to_forward()
        servo.cleanup()
        return
    finally:
        servo.rotate_dispenser = _original_rotate

    # ── Step 6: Tray tilt ─────────────────────────────────────────────────────
    print("─" * 60)
    print("STEP 6 — Tray tilt")
    print("When prompted, clear the tray to simulate pills falling into cup.")
    print("Press Enter to begin tray tilt sequence…")
    input()

    # Patch tray tilt servo to pause before post-tilt capture.
    _original_set_angle = servo.set_servo_angle

    tilt_call = [0]

    def _patched_set_angle(channel, angle):
        _original_set_angle(channel, angle)
        if channel == _cfg.SPECIAL_SERVO_CHANNELS[_cfg.TRAY_TILT_SERVO_INDEX] and angle > 0:
            tilt_call[0] += 1
            print(f"  [SERVO] Tray tilted to {angle}°. Clear the tray now, then press Enter…")
            input()

    servo.set_servo_angle = _patched_set_angle

    try:
        tilt_outcome = detector.tilt_tray(RESIDENT_ID)
        print(f"  ✓ Tray cleared after {tilt_outcome['attempts']} tilt(s).\n")
    except TrayTiltFailureError as e:
        print(f"  ✗ {e}")
        print("  → In production: alert staff.")
        tilt_outcome = {"tray_cleared": False, "error": str(e)}
    finally:
        servo.set_servo_angle = _original_set_angle

    # ── Step 7: User confirmation ─────────────────────────────────────────────
    print("─" * 60)
    print("STEP 7 — User confirmation")
    print("  [1] Confirmed")
    print("  [2] Discrepancy")
    print("  [3] Timeout")
    choice = input("Select (1/2/3): ").strip()
    status_map = {"1": "confirmed", "2": "discrepancy", "3": "timeout"}
    conf_status = status_map.get(choice, "timeout")
    detector.log_user_confirmation(RESIDENT_ID, conf_status)
    print(f"  Logged: {conf_status}\n")

    # ── Step 8: Session complete ──────────────────────────────────────────────
    print("─" * 60)
    print("STEP 8 — Session complete")
    complete = detector.log_session_complete(
        RESIDENT_ID, PRESCRIPTION_ID, summary, tilt_outcome, conf_status
    )
    print(f"  Session logged. eMAR flagged: {complete['emar_flagged']}")
    print(f"  Total audit events this session: {len(detector.get_audit_log())}")

    servo.cleanup()
    print("\n=== TEST COMPLETE ===\n")


# ── Entry point ───────────────────────────────────────────────────────────────

MODES = {"preview": mode_preview, "count": mode_count, "flow": mode_flow}

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode not in MODES:
        print("Usage: python test_pill_detection.py [preview | count | flow]")
        print()
        print("  preview  — live camera feed with circle-detection overlay")
        print("  count    — single capture, count, and save audit image")
        print("  flow     — full simulated dispense session (interactive)")
        sys.exit(1)
    MODES[mode]()
