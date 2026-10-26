"""
Facial Emotion Recognition Using Image Processing
Web interface (Streamlit) - image upload or camera snapshot.

Pipeline:  Input image -> Face detection (OpenCV) -> Face cropping/preprocessing
           -> CNN emotion model (pre-trained, via DeepFace) -> Label drawn on image
"""
import sys
import time

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image
from streamlit_back_camera_input import back_camera_input

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
COLORS = {  # BGR colours for the boxes
    "happy": (80, 200, 80), "sad": (180, 120, 60), "angry": (60, 60, 220),
    "fear": (170, 90, 150), "surprise": (200, 180, 0), "disgust": (90, 160, 90),
    "neutral": (150, 150, 150),
}


def analyse(bgr_image):
    """Detect every face and predict its emotion. Returns a list of dicts."""
    try:
        results = DeepFace.analyze(
            bgr_image,
            actions=["emotion"],
            detector_backend="opencv",   # OpenCV Haar-cascade face detector
            enforce_detection=True,      # raises ValueError if no face is found
            silent=True,
        )
    except ValueError:
        return []
    if isinstance(results, dict):        # older DeepFace versions return one dict
        results = [results]
    # number the faces from left to right so the list matches the picture
    return sorted(results, key=lambda r: r["region"]["x"])


def _overlap(a, b):
    """True if rectangles a and b (x1, y1, x2, y2) overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw(bgr_image, results):
    """Draw a box + label for every face. Labels are placed so that they never
    overlap each other (they move above / below / further out if needed)."""
    out = bgr_image.copy()
    H, W = out.shape[:2]
    faces = [(r["region"]["x"], r["region"]["y"],
              r["region"]["x"] + r["region"]["w"], r["region"]["y"] + r["region"]["h"])
             for r in results]
    placed = []                                    # label rectangles already used

    for n, (r, (x1, y1, x2, y2)) in enumerate(zip(results, faces), 1):
        label = r["dominant_emotion"]
        color = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{n} {label.capitalize()} {r['emotion'][label]:.0f}%"
        scale = max(0.4, min(0.7, (x2 - x1) / 150))        # small faces -> small text
        thick = 1 if scale < 0.55 else 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        lw, lh = tw + 8, th + 10
        lx = int(min(max(0, x1), max(0, W - lw)))           # keep inside the image

        # candidate label positions: above, below, then stepping further out
        cands = [y1 - lh, y2]
        for k in range(1, 6):
            cands += [y1 - lh - k * lh, y2 + k * lh]
        cands.append(y1)                                    # last resort: inside the box

        def free(rect, avoid_faces):
            if rect[1] < 0 or rect[3] > H:
                return False
            if any(_overlap(rect, p) for p in placed):
                return False
            if avoid_faces and any(_overlap(rect, f) for f in faces if f != (x1, y1, x2, y2)):
                return False
            return True

        chosen = None
        for avoid_faces in (True, False):                   # prefer not covering other faces
            for ly in cands:
                rect = (lx, int(ly), lx + lw, int(ly) + lh)
                if free(rect, avoid_faces):
                    chosen = rect
                    break
            if chosen:
                break
        if chosen is None:
            chosen = (lx, int(min(max(0, y1 - lh), H - lh)), lx + lw,
                      int(min(max(0, y1 - lh), H - lh)) + lh)
        placed.append(chosen)

        cv2.rectangle(out, (chosen[0], chosen[1]), (chosen[2], chosen[3]), color, -1)
        cv2.putText(out, text, (chosen[0] + 4, chosen[3] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)
        # thin line joining a moved label to its face
        if chosen[3] < y1 - 2 or chosen[1] > y2 + 2:
            cx = (x1 + x2) // 2
            ly = chosen[3] if chosen[3] <= y1 else chosen[1]
            ey = y1 if chosen[3] <= y1 else y2
            cv2.line(out, ((chosen[0] + chosen[2]) // 2, ly), (cx, ey), color, 1, cv2.LINE_AA)
    return out


def load(file):
    img = Image.open(file).convert("RGB")
    img.thumbnail((1000, 1000))          # keep it fast
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


st.title("🙂 Facial Emotion Recognition Using Image Processing")
st.caption("Mini Project – detects faces and predicts one of 7 emotions: "
           + ", ".join(e.capitalize() for e in EMOTIONS))

M_UPLOAD = "Upload an image"
M_BROWSER = "Take a photo (Webcam)"
mode = st.sidebar.radio("Choose input", [M_UPLOAD, M_BROWSER])
st.sidebar.info("Running live in the Cloud. Local hardware PC options are hidden for cross-device web stability.")

bgr = None

if mode == M_UPLOAD:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"])
    if file is not None:
        bgr = load(file)

elif mode == M_BROWSER:
    st.write("📸 **Webcam Stream Viewfinder**")
    st.write("Tap directly on the live video stream below to take your high-res snapshot!")
    
    # Fixed: Removed the unsupported 'label' keyword argument
    file = back_camera_input()
    
    if file is not None:
        bgr = load(file)

if bgr is not None:
    with st.spinner("Detecting faces and predicting emotions..."):
        results = analyse(bgr)

    if not results:
        st.warning("No face detected. Try a clearer, front-facing photo with good light.")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input image")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.image(cv2.cvtColor(draw(bgr, results), cv2.COLOR_BGR2RGB),
                     caption=f"{len(results)} face(s) detected")
        with right:
            st.subheader("Detected emotions")
            for i, r in enumerate(results, 1):
                label = r["dominant_emotion"]
                st.markdown(f"### Face {i}: {label.capitalize()} – {r['emotion'][label]:.0f}%")
elif mode == M_UPLOAD:
    st.info("Upload an image to begin.")
elif mode == M_BROWSER:
    st.info("Take a photo to begin.")
