#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import cv2

# ---- Optional: serial is only required if you actually connect Arduino ----
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

# ---------------- Paths (robust) ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_lbph_face_recognizer_model.yml"
CASCADE_PATH = ROOT / "models" / "haarcascade_frontalface_default.xml"

# ---------------- Sanity checks ----------------
if not MODEL_PATH.exists():
    sys.exit(f"[Error] Trained model not found at: {MODEL_PATH}")
if not CASCADE_PATH.exists():
    sys.exit(f"[Error] Haar cascade not found at: {CASCADE_PATH}")

if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    sys.exit("[Error] OpenCV contrib missing. Install: pip install opencv-contrib-python")

# ---------------- Config ----------------
# LBPH returns a distance (lower is better). Tune threshold for your dataset:
THRESHOLD = 60.0

# Who is allowed to trigger access (IDs → names)
ID_TO_NAME = {
    0: "pseudo",
    1: "Gabriel",
    2: "Dalyoung",
    3: "Godwill",
    4: "Bright",
    5: "Bena",
    6: "Sugira",
}
ALLOWLIST = {"Sugira"}  # only Sugira opens the gate/LED in this example

# Detection parameters
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_SIZE = (80, 80)

# Drawing
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
nametagColor = (255, 0, 0)
nametagHeight = 50
faceRectColor = nametagColor
faceRectThickness = 2

# Serial behavior
DEFAULT_BAUD = 9600
RESEND_INTERVAL_SEC = 5.0  # periodically re-send the current state

# ---------------- Helpers ----------------
def open_serial_port(preferred_port: str | None = None, baud: int = DEFAULT_BAUD):
    """
    Best effort: use preferred port if provided; else auto-pick a likely Arduino port on macOS.
    Returns a serial.Serial or None (dry mode).
    """
    if serial is None:
        print("[Info] pyserial not installed. Running in DRY MODE (no Arduino).")
        return None

    try:
        if preferred_port:
            s = serial.Serial(preferred_port, baud, timeout=1)
            time.sleep(2)  # Arduino reset
            print(f"[OK] Serial connected on {preferred_port} @ {baud}")
            return s

        # Try to auto-detect
        if list_ports is None:
            print("[Warn] list_ports unavailable. Provide a port manually.")
            return None

        candidates = list(list_ports.comports())
        # Prefer 'usbmodem' or 'usbserial' on macOS
        ordered = sorted(
            candidates,
            key=lambda p: (
                0 if ("usbmodem" in p.device or "usbserial" in p.device) else 1,
                p.device,
            ),
        )
        for p in ordered:
            try:
                s = serial.Serial(p.device, baud, timeout=1)
                time.sleep(2)
                print(f"[OK] Serial auto-connected on {p.device} @ {baud}")
                return s
            except Exception:
                continue
        print("[Warn] No Arduino serial port found. Running in DRY MODE.")
        return None
    except Exception as e:
        print(f"[Warn] Could not open serial port: {e}. DRY MODE.")
        return None

def send_to_arduino(ser, value: bytes, *, force: bool = False):
    """
    Send a single byte (b'0' or b'1') to Arduino.
    If ser is None, print instead (dry mode).
    """
    if ser is None:
        print(f"[DRY] -> Arduino: {value!r}")
        return
    try:
        ser.write(value)
        # Optional: ser.flush()  # rarely needed at 9600
    except Exception as e:
        print(f"[Warn] Serial write failed: {e}")

# ---------------- Load models ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

faceCascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if faceCascade.empty():
    sys.exit(f"[Error] Failed to load cascade: {CASCADE_PATH}")

# ---------------- Camera ----------------
# macOS: try AVFoundation first
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[Error] Could not open camera. Check macOS camera permissions.")

# ---------------- Serial ----------------
# Change this to your known port or leave None for auto-detect:
PREFERRED_PORT = None  # e.g., "/dev/tty.usbmodem2201"
ser = open_serial_port(PREFERRED_PORT, DEFAULT_BAUD)

# ---------------- Loop ----------------
last_sent = None          # last byte actually sent to Arduino (b'0' or b'1')
last_change_ts = 0.0      # last time we resent the current state (for keep-alive)
access_state = b'0'       # current desired state

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Warn] Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        is_access_granted = False

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            pred_id, dist = recognizer.predict(roi)

            # Accept match if LBPH distance is within threshold
            is_match = dist <= THRESHOLD
            name = ID_TO_NAME.get(pred_id, f"ID:{pred_id}") if is_match else "Unknown"

            if is_match and name in ALLOWLIST:
                is_access_granted = True

            # Visuals
            cv2.rectangle(frame, (x, y), (x + w, y + h), faceRectColor, faceRectThickness)
            # Friendly "confidence %": map [0, THRESHOLD] → [100, 0]
            conf_pct = max(0, 100 - (dist / THRESHOLD) * 100) if is_match else 0.0

            top = max(0, y - nametagHeight - 10)
            cv2.rectangle(frame, (x - 2, top), (x + w + 2, top + nametagHeight), nametagColor, -1)
            label = f"{name}: {conf_pct:.1f}%  (d={dist:.1f})"
            cv2.putText(frame, label, (x + 4, top + nametagHeight - 15), fontFace, fontScale, fontColor, fontWeight)

        # Decide output state
        access_state = b'1' if is_access_granted else b'0'

        # Debounce: send only on change, or periodically as keep-alive
        now = time.time()
        if access_state != last_sent or (now - last_change_ts) > RESEND_INTERVAL_SEC:
            send_to_arduino(ser, access_state)
            last_sent = access_state
            last_change_ts = now

        # UI
        cv2.putText(
            frame,
            f"Faces: {len(faces)} | Access: {'GRANTED' if access_state == b'1' else 'DENIED'}",
            (10, 25),
            fontFace, 0.6,
            (0, 255, 0) if access_state == b'1' else (0, 0, 255),
            2
        )

        cv2.imshow("Face Recognition + Arduino Control", frame)
        if cv2.waitKey(1) & 0xFF == 'q':
            break

finally:
    # Always leave the device OFF on exit
    try:
        if last_sent != b'0':
            send_to_arduino(ser, b'0', force=True)
    except Exception:
        pass

    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()