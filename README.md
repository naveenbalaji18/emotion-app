# Facial Emotion Recognition – how to run (no coding needed)

## One-time setup
1. Install **Python 3.10 or 3.11** from python.org (tick "Add Python to PATH").
2. Put this folder somewhere easy, e.g. Desktop/emotion_app.
3. Open a terminal in the folder (Windows: type `cmd` in the folder's address bar).
4. Run:
   python -m venv venv
   venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   (takes a few minutes; needs internet)

## Run the web app (image upload / camera photo)
   streamlit run app.py
A browser tab opens. The first prediction downloads the model (internet needed, one time).

## Run real-time multi-face webcam
   python live_webcam.py        (press q to quit)

## If something fails
- "No module named ..." -> activate the venv again, then pip install -r requirements.txt
- Slow first run -> the model is downloading; wait.
- No face detected -> use a front-facing, well-lit photo.
