"""
electronic/pill_detection.py
PillDetector — camera-based pill presence and count verification.

Phase 1 scope: pill PRESENCE and COUNT only.
No colour, shape, or type verification at this stage.

Integrates with:
  - config.hardware_config   (angles, delays, paths)
  - electronic.servo_controller.ServoController  (camera + tray servos)
  - OpenCV / shared camera   (image capture and pill counting)

Typical call sequence (from main flow):
    detector = PillDetector(servo_controller)

    # Step 4 — tray pre-check
    detector.check_tray_empty(resident_id)

    # Step 5 — dispense + verify each pill
    summary = detector.run_dispense_session(
        resident_id, prescription_id, prescription
    )

    # Step 6 — tray tilt → camera returns to forward inside this call
    tilt_outcome = detector.tilt_tray(resident_id)

    # Step 7 — log confirmation result
    detector.log_user_confirmation(resident_id, status="confirmed")

    # Step 8 — log session complete
    detector.log_session_complete(resident_id, prescription_id,
                                  summary, tilt_outcome, "confirmed")
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from config.hardware_config import (
    AUDIT_IMAGE_DIR,
    CAMERA_RETURN_ANGLE,
    CAMERA_SETTLE_DELAY,
    CAMERA_TILT_ANGLE,
    CAMERA_TILT_SERVO_INDEX,
    CONFIRM_TIMEOUT,  # noqa: F401 — re-exported for callers
    MAX_DISPENSE_ATTEMPTS,
    MAX_TILT_ATTEMPTS,
    PILL_SETTLE_DELAY,
    SPECIAL_SERVO_CHANNELS,
    TRAY_TILT_ANGLE,
    TRAY_TILT_DURATION,
    TRAY_TILT_SERVO_INDEX,
)
from electronic.servo_controller import ServoController

# Camera index — same physical camera used by facial_recognition.py.
# The camera_tilt_servo switches it between forward-facing and downward.
_CAMERA_INDEX = 1

# ── Hough Circle Transform tuning parameters ──────────────────────────────────
# Adjust these during on-device calibration once the tray camera angle is fixed.
# Phase 1: presence + count only; no shape/colour classification.
_HOUGH_DP         = 1    # inverse ratio of accumulator resolution to image resolution
_HOUGH_MIN_DIST   = 20   # minimum distance (px) between detected circle centres
_HOUGH_PARAM1     = 50   # Canny edge upper threshold
_HOUGH_PARAM2     = 30   # accumulator threshold — lower = more (possibly false) detections
_HOUGH_MIN_RADIUS = 5    # minimum circle radius in pixels
_HOUGH_MAX_RADIUS = 50   # maximum circle radius in pixels

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class TrayNotClearError(Exception):
    """Raised by check_tray_empty() when pills are present on the tray."""


class DispenseFailureError(Exception):
    """Raised by run_dispense_session() when a pill cannot be verified
    after MAX_DISPENSE_ATTEMPTS."""


class TrayTiltFailureError(Exception):
    """Raised by tilt_tray() when the tray is not cleared after
    MAX_TILT_ATTEMPTS tilts."""


# ── PillDetector ──────────────────────────────────────────────────────────────

class PillDetector:
    """
    Manages camera repositioning, pill counting, dispense verification,
    tray tilt, and structured audit logging for the PillWheel dispenser.

    All public methods append structured events to an in-memory session log
    (accessible via get_audit_log()) and emit them through the standard
    logging module at INFO level.

    The caller is responsible for:
      - Facial recognition (Step 3) — done while camera is forward-facing.
      - Displaying on-screen alerts returned in exception messages.
      - Waiting for staff acknowledgement and calling reset_session()
        before starting a new dispense session.
    """

    def __init__(self, servo_controller: ServoController):
        self._servo = servo_controller
        self._camera_is_down = False   # tracks current camera servo position
        self._session_log: list = []
        os.makedirs(AUDIT_IMAGE_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_tray_empty(self, resident_id: str) -> dict:
        """
        Tray pre-check (Step 4).

        Repositions camera downward, captures an image, and verifies 0 pills.

        Returns:
            dict — keys: event, resident_id, timestamp, pill_count,
                          tray_clear (True), image_path

        Raises:
            TrayNotClearError — if any pills are detected.
            Caller should display: "Tray not clear. Please contact a carer."
        """
        self._reposition_camera_down()
        image, image_path = self._capture_tray_image(
            resident_id=resident_id,
            medication_name="precheck",
            pill_index=0,
            attempt=0,
        )
        count = self._count_pills(image)
        event = {
            "event":       "tray_precheck",
            "resident_id": resident_id,
            "timestamp":   _now(),
            "pill_count":  count,
            "tray_clear":  count == 0,
            "image_path":  image_path,
        }
        self._log(event)

        if count != 0:
            logger.warning(
                "Tray pre-check FAILED — %d pill(s) detected. Halting.", count
            )
            raise TrayNotClearError(
                f"Tray not clear — {count} pill(s) detected before dispensing."
            )

        logger.info("Tray pre-check PASSED — tray is empty.")
        return event

    def run_dispense_session(
        self,
        resident_id: str,
        prescription_id: str,
        prescription: list,
    ) -> dict:
        """
        Dispense + verify each pill sequentially (Step 5).

        prescription format:
            [
                {"medication_name": "Vitamin D", "dispenser_index": 0},
                {"medication_name": "Vitamin C", "dispenser_index": 1},
                ...
            ]

        Pills are dispensed one at a time.  After each dispense the
        *cumulative* visible count is verified:
            after pill 0 → expect 1, after pill 1 → expect 2, etc.

        Returns:
            dict — session summary with per-pill dispense records.

        Raises:
            DispenseFailureError — if any pill exceeds MAX_DISPENSE_ATTEMPTS.
            Caller should display:
                "Unable to dispense medication. A carer has been notified.
                 Please wait."
            Machine must be locked until staff manually resets.
        """
        self._log({
            "event":           "session_start",
            "resident_id":     resident_id,
            "prescription_id": prescription_id,
            "timestamp":       _now(),
        })

        dispense_records: list = []
        successfully_dispensed = 0   # cumulative verified pill count

        for pill_index, pill in enumerate(prescription):
            med_name       = pill["medication_name"]
            dispenser_idx  = pill["dispenser_index"]
            expected_count = successfully_dispensed + 1
            pill_verified  = False

            for attempt in range(1, MAX_DISPENSE_ATTEMPTS + 1):
                logger.info(
                    "Pill %d (%s) — dispense attempt %d/%d",
                    pill_index, med_name, attempt, MAX_DISPENSE_ATTEMPTS,
                )

                # 1. Trigger dispense servo
                self._servo.rotate_dispenser(dispenser_idx)

                # 2. Wait for pill to settle
                time.sleep(PILL_SETTLE_DELAY)

                # 3 + 4. Capture image and count
                image, image_path = self._capture_tray_image(
                    resident_id, med_name, pill_index, attempt
                )
                detected = self._count_pills(image)

                # 6. Save audit record regardless of outcome
                record = {
                    "event":           "dispense_attempt",
                    "resident_id":     resident_id,
                    "prescription_id": prescription_id,
                    "pill_index":      pill_index,
                    "medication_name": med_name,
                    "attempt":         attempt,
                    "detected_count":  detected,
                    "expected_count":  expected_count,
                    "timestamp":       _now(),
                    "image_path":      image_path,
                    "success":         detected == expected_count,
                }
                self._log(record)
                dispense_records.append(record)

                # 7. Success
                if detected == expected_count:
                    logger.info(
                        "Pill %d VERIFIED — detected=%d, expected=%d.",
                        pill_index, detected, expected_count,
                    )
                    successfully_dispensed += 1
                    pill_verified = True
                    break

                # 8. Mismatch — retry
                logger.warning(
                    "Pill %d mismatch — detected=%d, expected=%d. Retrying.",
                    pill_index, detected, expected_count,
                )

            # 9. Exhausted all attempts for this pill
            if not pill_verified:
                failure_event = {
                    "event":              "dispense_failure",
                    "resident_id":        resident_id,
                    "prescription_id":    prescription_id,
                    "pill_index":         pill_index,
                    "medication_name":    med_name,
                    "attempts_exhausted": MAX_DISPENSE_ATTEMPTS,
                    "timestamp":          _now(),
                    "image_path":         dispense_records[-1]["image_path"],
                }
                self._log(failure_event)
                raise DispenseFailureError(
                    f"Unable to dispense pill {pill_index} ({med_name}) "
                    f"after {MAX_DISPENSE_ATTEMPTS} attempts."
                )

        summary = {
            "event":                  "dispense_session_complete",
            "resident_id":            resident_id,
            "prescription_id":        prescription_id,
            "total_pills":            len(prescription),
            "successfully_dispensed": successfully_dispensed,
            "dispense_records":       dispense_records,
            "timestamp":              _now(),
        }
        self._log(summary)
        return summary

    def tilt_tray(self, resident_id: str) -> dict:
        """
        Tilt tray to funnel pills into cup (Step 6).

        Tilts the tray servo, waits, captures an image, and verifies the
        tray is empty.  Re-tilts up to MAX_TILT_ATTEMPTS on failure.
        Returns camera to forward-facing position before returning.

        Returns:
            dict — keys: event, tray_cleared (True), attempts, image_paths,
                          timestamp

        Raises:
            TrayTiltFailureError — if tray is not empty after MAX_TILT_ATTEMPTS.
            Caller should display:
                "Please check your cup. A carer has been notified."
        """
        tray_channel   = SPECIAL_SERVO_CHANNELS[TRAY_TILT_SERVO_INDEX]
        image_paths: list = []

        try:
            for attempt in range(1, MAX_TILT_ATTEMPTS + 1):
                logger.info("Tray tilt attempt %d/%d.", attempt, MAX_TILT_ATTEMPTS)

                # 1. Tilt tray
                self._servo.set_servo_angle(tray_channel, TRAY_TILT_ANGLE)
                self._log({
                    "event":     "tray_tilt_executed",
                    "attempt":   attempt,
                    "angle":     TRAY_TILT_ANGLE,
                    "timestamp": _now(),
                })

                # 2. Wait for pills to funnel into cup
                time.sleep(TRAY_TILT_DURATION)

                # 3. Capture image
                image, image_path = self._capture_tray_image(
                    resident_id=resident_id,
                    medication_name="traytilt",
                    pill_index=attempt,
                    attempt=attempt,
                )
                image_paths.append(image_path)

                # 4. Check if tray is empty
                count     = self._count_pills(image)
                tray_empty = count == 0
                self._log({
                    "event":       "tray_tilt_check",
                    "attempt":     attempt,
                    "pill_count":  count,
                    "tray_empty":  tray_empty,
                    "image_path":  image_path,
                    "timestamp":   _now(),
                })

                # 5. Return tray to flat
                self._servo.set_servo_angle(tray_channel, 0)

                if tray_empty:
                    logger.info("Tray clear after tilt attempt %d.", attempt)
                    break
            else:
                # All tilt attempts exhausted without clearing the tray
                failure_event = {
                    "event":         "tray_clearance_failure",
                    "resident_id":   resident_id,
                    "attempts":      MAX_TILT_ATTEMPTS,
                    "image_paths":   image_paths,
                    "staff_alerted": True,
                    "timestamp":     _now(),
                }
                self._log(failure_event)
                raise TrayTiltFailureError(
                    f"Tray not clear after {MAX_TILT_ATTEMPTS} tilt attempts."
                )

        finally:
            # 6. Always return camera to forward-facing position
            self._reposition_camera_forward()

        return {
            "event":        "tray_tilt_success",
            "tray_cleared": True,
            "attempts":     attempt,
            "image_paths":  image_paths,
            "timestamp":    _now(),
        }

    def return_camera_to_forward(self):
        """
        Explicitly return camera to forward-facing position.

        Call this in error-handling paths (e.g. after TrayNotClearError)
        so the screen is usable for staff interaction.
        """
        self._reposition_camera_forward()

    def log_user_confirmation(
        self,
        resident_id: str,
        status: str,
        image_refs: Optional[list] = None,
    ) -> dict:
        """
        Log the outcome of the user confirmation step (Step 7).

        status — one of: "confirmed" | "discrepancy" | "timeout"
        image_refs — list of audit image paths to attach to discrepancy events.
        """
        event = {
            "event":       "user_confirmation",
            "resident_id": resident_id,
            "status":      status,
            "image_refs":  image_refs or [],
            "timestamp":   _now(),
        }
        self._log(event)
        return event

    def log_session_complete(
        self,
        resident_id: str,
        prescription_id: str,
        dispense_summary: dict,
        tilt_outcome: dict,
        confirmation_status: str,
    ) -> dict:
        """
        Log the full session summary (Step 8).

        Returns the summary dict (also appended to the audit log).
        Flagged for future eMAR integration.
        """
        event = {
            "event":               "session_complete",
            "resident_id":         resident_id,
            "prescription_id":     prescription_id,
            "dispense_summary":    dispense_summary,
            "tray_tilt_outcome":   tilt_outcome,
            "confirmation_status": confirmation_status,
            "emar_flagged":        True,
            "full_audit_log":      self.get_audit_log(),
            "timestamp":           _now(),
        }
        self._log(event)
        return event

    def get_audit_log(self) -> list:
        """Return a copy of this session's in-memory audit log."""
        return list(self._session_log)

    def reset_session(self):
        """
        Clear the in-memory session log.
        Call this after staff acknowledgement, before starting a new session.
        """
        self._session_log = []
        logger.info("PillDetector session reset.")

    # ── Camera control ────────────────────────────────────────────────────────

    def _reposition_camera_down(self):
        """Tilt camera servo to downward pill-detection position."""
        if self._camera_is_down:
            return
        camera_channel = SPECIAL_SERVO_CHANNELS[CAMERA_TILT_SERVO_INDEX]
        self._servo.set_servo_angle(camera_channel, CAMERA_TILT_ANGLE)
        time.sleep(CAMERA_SETTLE_DELAY)
        self._camera_is_down = True
        self._log({
            "event":     "camera_reposition",
            "direction": "down",
            "angle":     CAMERA_TILT_ANGLE,
            "timestamp": _now(),
        })
        logger.info("Camera repositioned downward (%.1f deg).", CAMERA_TILT_ANGLE)

    def _reposition_camera_forward(self):
        """Return camera servo to forward-facing position."""
        if not self._camera_is_down:
            return
        camera_channel = SPECIAL_SERVO_CHANNELS[CAMERA_TILT_SERVO_INDEX]
        self._servo.set_servo_angle(camera_channel, CAMERA_RETURN_ANGLE)
        time.sleep(CAMERA_SETTLE_DELAY)
        self._camera_is_down = False
        self._log({
            "event":     "camera_reposition",
            "direction": "forward",
            "angle":     CAMERA_RETURN_ANGLE,
            "timestamp": _now(),
        })
        logger.info("Camera returned to forward position (%.1f deg).", CAMERA_RETURN_ANGLE)

    # ── Image capture ─────────────────────────────────────────────────────────

    def _capture_tray_image(
        self,
        resident_id: str,
        medication_name: str,
        pill_index: int,
        attempt: int,
    ) -> tuple:
        """
        Reposition camera downward if needed, capture one frame, and save
        it to AUDIT_IMAGE_DIR.

        Filename format:
            {resident_id}_{medication_name}_{pill_index}_{timestamp}.jpg

        Returns:
            (bgr_image, image_path) — bgr_image is a numpy array (blank
            zeros array if capture fails, so audit trail is never broken).
        """
        if not self._camera_is_down:
            self._reposition_camera_down()

        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_med = medication_name.replace(" ", "_")
        filename = f"{resident_id}_{safe_med}_{pill_index}_{ts_str}.jpg"
        image_path = os.path.join(AUDIT_IMAGE_DIR, filename)

        cap   = cv2.VideoCapture(_CAMERA_INDEX)
        image = None
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(image_path, frame)
            image = frame
            logger.info("Audit image saved: %s", image_path)
        else:
            logger.error(
                "Camera capture failed (index %d). Writing blank placeholder.",
                _CAMERA_INDEX,
            )
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(image_path, image)

        return image, image_path

    # ── Pill counting ─────────────────────────────────────────────────────────

    def _count_pills(self, image: Optional[np.ndarray]) -> int:
        """
        Count pills in *image* using the Hough Circle Transform.

        Phase 1: presence and count only — no colour or shape classification.
        Detection parameters are module-level constants (_HOUGH_*) and can
        be tuned after on-device calibration without changing this logic.

        Returns 0 if image is None or if no circles are found.
        """
        if image is None:
            return 0

        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=_HOUGH_DP,
            minDist=_HOUGH_MIN_DIST,
            param1=_HOUGH_PARAM1,
            param2=_HOUGH_PARAM2,
            minRadius=_HOUGH_MIN_RADIUS,
            maxRadius=_HOUGH_MAX_RADIUS,
        )

        count = 0 if circles is None else len(circles[0])
        logger.debug("Pill count (Hough circles): %d", count)
        return count

    # ── Audit logging ─────────────────────────────────────────────────────────

    def _log(self, event: dict):
        """Append *event* to the in-memory session log and emit via logger."""
        self._session_log.append(event)
        logger.info("[AUDIT] %s", json.dumps(event, default=str))


# ── Module helper ─────────────────────────────────────────────────────────────

def _now() -> str:
    """Return the current local time as an ISO-8601 string (second precision)."""
    return datetime.now().isoformat(timespec="seconds")
