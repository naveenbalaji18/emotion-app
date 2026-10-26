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

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        results = DeepFace.analyze(img, actions=["emotion"], detector_backend="opencv", enforce_detection=False, silent=True)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            x, y, w, h = r["region"]["x"], r["region"]["y"], r["region"]["w"], r["region"]["h"]
            label = r["dominant_emotion"]
            color = COLORS.get(label, (0, 255, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label.capitalize()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception:
        pass
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🙂 Custom Camera Selector - Emotion Detector")
mode = st.sidebar.radio("Choose input", ["Static Image Upload", "Interactive Real-Time Viewfinder"])

if mode == "Static Image Upload":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        st.image(img, caption="Uploaded Image")
else:
    st.write("🔄 **Use the camera device dropdown box directly underneath the video to change lenses manually!**")
    
    # Updated: Explicitly telling the WebRTC UI engine to render device pickers 
    # and setting video constraint defaults so mobile browsers expose all tracks.
    webrtc_streamer(
        key="emotion-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "facingMode": "environment"  # Starts on back camera by default
            },
            "audio": False
        },
        async_processing=True,
    )
