"""
Facial Emotion Recognition Using Image Processing
Web interface (Streamlit) - image upload or camera snapshot.

Pipeline:  Input image -> Face detection (OpenCV) -> Face cropping/preprocessing
           -> CNN emotion model (pre-trained, via DeepFace) -> Label drawn on image
"""
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

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


CAM_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def grab_frame(index):
    """Take one picture from camera number `index` (None if it cannot be used)."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, CAM_BACKEND)
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(10):          # skip a few frames so brightness settles
            ok, f = cap.read()
            if ok:
                frame = f
        return frame
    except cv2.error:
        return None
    finally:
        if cap is not None:
            cap.release()


def detect_cameras(max_index=5):
    """Return the list of camera numbers that open on this computer."""
    found = []
    for i in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(i, CAM_BACKEND)
            if cap.isOpened():
                found.append(i)
        except cv2.error:
            pass
        finally:
            if cap is not None:
                cap.release()
    return found or [0]


st.title("🙂 Facial Emotion Recognition Using Image Processing")
st.caption("Mini Project – detects faces and predicts one of 7 emotions: "
           + ", ".join(e.capitalize() for e in EMOTIONS))

M_UPLOAD = "Upload an image"
M_BROWSER = "Take a photo (browser camera)"
M_PC = "Take a photo (choose PC camera)"
# On Streamlit Community Cloud the app runs on a remote server that has no camera,
# so the "PC camera" mode can only be offered when the app runs on your own computer.
IS_CLOUD = os.path.exists("/mount/src") or os.environ.get("IS_CLOUD") == "1"
modes = [M_UPLOAD, M_BROWSER] if IS_CLOUD else [M_UPLOAD, M_BROWSER, M_PC]
mode = st.sidebar.radio("Choose input", modes)
if IS_CLOUD:
    st.sidebar.caption("Camera mode uses your own device's camera through the browser. "
                       "Allow camera permission when the browser asks.")
st.sidebar.info("For continuous real-time webcam tracking, run `live_webcam.py` "
                "(see README).\n\n**On a phone:** choose 'Upload an image' and tap "
                "'Take photo' / 'Camera' in the file picker.")


def take_photo():
    """Button callback: use the latest viewfinder frame (the camera is already open),
    or open the camera once if the viewfinder is off."""
    idx = st.session_state.get("cam_sel", 0)
    frame = None
    last = st.session_state.get("last_frame")
    if st.session_state.get("live") and last is not None and last[0] == idx:
        frame = last[1]
    if frame is None:
        frame = grab_frame(idx)
    if frame is None:
        st.session_state["cam_err"] = True
    else:
        st.session_state["cam_err"] = False
        st.session_state["shot"] = frame
        st.session_state["live"] = False      # stop the viewfinder to show the result


bgr = None
view = st.empty()                      # the live viewfinder is drawn here
if mode == M_UPLOAD:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"])
    if file is not None:
        bgr = load(file)
elif mode == M_BROWSER:
    file = st.camera_input("Look at the camera and click 'Take photo'")
    if file is not None:
        bgr = load(file)
else:
    cams = st.session_state.get("cams", [0, 1, 2, 3])
    c1, c2, c3 = st.columns([2, 1, 1])
    cam = c1.selectbox("Select camera", cams, key="cam_sel", format_func=lambda i: f"Camera {i}")
    c2.write("")
    if c2.button("Detect cameras"):
        with st.spinner("Looking for cameras..."):
            st.session_state["cams"] = detect_cameras()
        st.rerun()
    c3.write("")
    c3.button("📷 Take photo", type="primary", on_click=take_photo)
    st.toggle("Show live viewfinder", value=True, key="live")
    if st.session_state.get("cam_err"):
        st.error(f"Camera {cam} could not be opened. Choose another camera number, "
                 "or close other apps/tabs that are using the camera.")
    st.caption("Camera 0 is usually the laptop webcam; an external camera is usually 1 or 2. "
               "This mode uses the cameras of the computer running the app. "
               "On a phone, use 'Upload an image' instead.")
    bgr = st.session_state.get("shot")

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
else:
    st.info("Choose a camera and click 'Take photo' to begin.")

# ---- live viewfinder for the PC-camera mode (runs last, until you click something) ----
if mode == M_PC and st.session_state.get("live"):
    idx = st.session_state.get("cam_sel", 0)
    cap = None
    try:
        cap = cv2.VideoCapture(idx, CAM_BACKEND)
        if not cap.isOpened():
            view.error("This camera could not be opened. Choose another camera number.")
        else:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                st.session_state["last_frame"] = (idx, f)
                view.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), caption="Live viewfinder", width=640)
                time.sleep(0.03)
    except cv2.error:
        view.error("The camera stopped responding. Choose another camera number or retry.")
    finally:
        if cap is not None:
            cap.release()
