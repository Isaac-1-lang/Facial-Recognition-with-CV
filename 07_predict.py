#!/usr/bin/env python3
import cv2
import sys
from pathlib import Path
import time

# -------- Paths (robust to where you run it) --------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_lbph_face_recognizer_model.yml"
CASCADE_PATH = ROOT / "models" / "haarcascade_frontalface_default.xml"

# -------- Sanity checks --------
if not MODEL_PATH.exists():
    sys.exit(f"[Error] Trained model not found at: {MODEL_PATH}")
if not CASCADE_PATH.exists():
    sys.exit(f"[Error] Haar cascade not found at: {CASCADE_PATH}")

# -------- Load recognizer & cascade --------
if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    sys.exit("[Error] OpenCV contrib missing. Install opencv-contrib-python.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

faceCascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if faceCascade.empty():
    sys.exit(f"[Error] Failed to load cascade: {CASCADE_PATH}")

# -------- Name mapping (edit as needed) --------
ID_TO_NAME = {
    0: "pseudo",
    1: "Gabriel",
    2: "Dalyoung",
    3: "Godwill",
    4: "Bright",
    5: "Bena",
    6: "Sugira",
}

# -------- Drawing styles --------
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

nametagColor = (255, 0, 0)
nametagHeight = 50

faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

# -------- Detection params (tweak here) --------
SCALE_FACTOR = 1.2       # 1.1–1.3; higher skips more scales (faster, sometimes fewer detections)
MIN_NEIGHBORS = 5        # 3–6; higher → fewer false positives
MIN_SIZE = (80, 80)      # increase if you only want bigger faces

# -------- Recognition threshold --------
# LBPH returns a distance; LOWER is better. Typical acceptance: <= 60 (tune by your data).
THRESHOLD = 60.0

# -------- Camera (macOS backend helps) --------
# Try AVFoundation on macOS; fallback to default if it fails.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[Error] Could not open camera. Check permissions (System Settings → Privacy & Security → Camera).")

prev = time.time(); fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[Warn] Failed to read frame.")
        break

    # Grayscale + histogram equalization for better detection in poor lighting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]

        # Predict (LBPH distance)
        pred_id, dist = recognizer.predict(roi)

        # Decision: accept if distance <= THRESHOLD, else Unknown
        is_match = dist <= THRESHOLD
        name = ID_TO_NAME.get(pred_id, f"ID:{pred_id}") if is_match else "Unknown"

        # For display, compute a friendly "confidence %" (0–100 where higher is better)
        # Map dist in [0, THRESHOLD] to [100, 0], clip at 0
        conf_pct = max(0, 100 - (dist / THRESHOLD) * 100)

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), faceRectangleBorderColor, faceRectangleBorderSize)

        # Name tag
        top = max(0, y - nametagHeight - 10)
        cv2.rectangle(frame, (x - 2, top), (x + w + 2, top + nametagHeight), nametagColor, -1)
        label = f"{name}: {conf_pct:.1f}%  (d={dist:.1f})"
        cv2.putText(frame, label, (x + 4, top + nametagHeight - 15), fontFace, fontScale, fontColor, fontWeight)

    # FPS overlay
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if prev else 0.0
    prev = now
    cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.1f}", (10, 25), fontFace, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection & Recognition (LBPH)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()