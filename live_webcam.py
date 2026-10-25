"""
Real-time multi-face emotion recognition from the webcam.
Run:  python live_webcam.py        Press  q  to quit.
"""
import cv2
from deepface import DeepFace

COLORS = {
    "happy": (80, 200, 80), "sad": (180, 120, 60), "angry": (60, 60, 220),
    "fear": (170, 90, 150), "surprise": (200, 180, 0), "disgust": (90, 160, 90),
    "neutral": (150, 150, 150),
}
ANALYSE_EVERY = 5      # run the model every 5th frame (keeps video smooth)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and retry.")

frame_no, last = 0, []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_no % ANALYSE_EVERY == 0:
        try:
            res = DeepFace.analyze(frame, actions=["emotion"],
                                   detector_backend="opencv",
                                   enforce_detection=True, silent=True)
            last = [res] if isinstance(res, dict) else res
        except ValueError:
            last = []                      # no face in this frame
    for r in last:                         # draw every face found (multi-face)
        b = r["region"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        label = r["dominant_emotion"]
        color = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {r['emotion'][label]:.0f}%", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Facial Emotion Recognition - press q to quit", frame)
    frame_no += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
