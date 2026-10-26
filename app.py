"""
Facial Emotion Recognition Using Image Processing
Web interface (Streamlit) - image upload or real-time camera selection.
"""
import av
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
COLORS = {
    "happy": (80, 200, 80), "sad": (180, 120, 60), "angry": (60, 60, 220),
    "fear": (170, 90, 150), "surprise": (200, 180, 0), "disgust": (90, 160, 90),
    "neutral": (150, 150, 150),
}

def analyse(bgr_image):
    """Detect faces and predict emotions."""
    try:
        results = DeepFace.analyze(
            bgr_image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return []
    if isinstance(results, dict):
        results = [results]
    return sorted(results, key=lambda r: r["region"]["x"])

def video_frame_callback(frame):
    """Callback engine to process live video frames and render overlay boxes."""
    img = frame.to_ndarray(format="bgr24")
    results = analyse(img)
    
    for n, r in enumerate(results, 1):
        x, y, w, h = r["region"]["x"], r["region"]["y"], r["region"]["w"], r["region"]["h"]
        label = r["dominant_emotion"]
        color = COLORS.get(label, (0, 255, 0))
        
        # Draw bounding boxes and text overlay flags
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{n} {label.capitalize()} {r['emotion'][label]:.0f}%"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🙂 Facial Emotion Recognition Using Image Processing")
st.caption("Mini Project – detects faces and predicts one of 7 emotions: " + ", ".join(e.capitalize() for e in EMOTIONS))

M_UPLOAD = "Upload an image"
M_LIVE = "Interactive Real-Time Viewfinder"
mode = st.sidebar.radio("Choose input", [M_UPLOAD, M_LIVE])

if mode == M_UPLOAD:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file is not None:
        img = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing uploaded image..."):
            results = analyse(bgr)
            
        if not results:
            st.warning("No face detected. Try a clearer photo with better lighting.")
            st.image(img, caption="Input image")
        else:
            # Draw on static photo
            for n, r in enumerate(results, 1):
                x, y, w, h = r["region"]["x"], r["region"]["y"], r["region"]["w"], r["region"]["h"]
                label = r["dominant_emotion"]
                color = COLORS.get(label, (0, 255, 0))
                cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(bgr, f"{n} {label.capitalize()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            left, right = st.columns([3, 2])
            with left:
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Analyzed Image")
            with right:
                st.subheader("Detected emotions")
                for i, r in enumerate(results, 1):
                    lbl = r["dominant_emotion"]
                    st.markdown(f"### Face {i}: {lbl.capitalize()} – {r['emotion'][lbl]:.0f}%")
else:
    st.write("🔄 **Use the video device selector dropdown directly below the player window to manually toggle between your front camera and separate rear high-res lenses!**")
    
    webrtc_streamer(
        key="emotion-live-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:://google.com"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
