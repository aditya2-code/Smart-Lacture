# Smart-Lecture 🎓
### Emotion-Aware Video Player — AI in Education

A Flask web app that watches your face while you watch a lecture, and **automatically pauses** when you look confused or drowsy.

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install flask opencv-python numpy
```

### 2. Run the app
```bash
python app.py
```

### 3. Open your browser
```
http://localhost:5000
```

### 4. Use the app
1. Allow webcam access when prompted
2. Wait 3 seconds for **calibration** (stay neutral, look at camera)
3. Click **"+ Load Lecture"** and pick any `.mp4` video file
4. Press play — the AI will watch your face in real-time

---

## 🧠 How it detects emotions

### Confusion Detection (Brow Furrow)
- MediaPipe/OpenCV detects eye position relative to the face
- Measures the **brow-to-eye distance ratio** against your calibrated baseline
- If your brows drop >18% from baseline for **2 seconds** → confusion triggered

### Drowsiness Detection (EAR — Eye Aspect Ratio)
- Industry-standard metric: `EAR = eye_height / eye_width`
- If EAR drops below **65% of your baseline** → drowsy state
- Sustained for **1.5 seconds** → quiz triggered

### Personal Calibration
- On startup, the app captures **30 frames** of your neutral face
- All thresholds are relative to **your** face — not a generic model
- Click **"Recalibrate"** anytime (e.g. if lighting changes)

---

## 📁 Project Structure
```
smart_lecture/
├── app.py                 # Flask backend + OpenCV emotion engine
├── requirements.txt
└── templates/
    └── index.html         # Full UI (video player + webcam panel)
```

---

## 🎮 Controls

| Action | How |
|--------|-----|
| Load video | Click "+ Load Lecture" (top right) |
| Play/Pause | Space bar or ▶ button |
| Skip ±10s | Click -10 / +10 buttons |
| Seek | Click anywhere on the progress bar |
| Recalibrate | Click "Recalibrate" (top bar) |
| Test without webcam | Use "Simulate" buttons in sidebar |

---

## 🔧 Troubleshooting

**No webcam feed?**
- The app falls back to **demo mode** — all features still work with simulated data
- In demo mode, use the "Simulate Confusion / Drowsiness" buttons in the sidebar

**False positives?**
- Click **Recalibrate** to reset your baseline
- Ensure consistent lighting on your face

**Video won't load?**
- Supported formats: MP4, WebM, MOV (anything your browser supports natively)
- No server upload needed — video plays directly from your local file

---

## 🚀 Extending the app

### Add real AI summaries
Replace the `SUMMARIES` list in `index.html` with an API call:
```python
# In app.py, add:
import anthropic
client = anthropic.Anthropic()

@app.route('/api/summary')
def get_summary():
    timestamp = request.args.get('t', '0')
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role":"user","content":f"Summarize what a professor might be teaching at timestamp {timestamp} in a linear algebra lecture."}]
    )
    return jsonify({"summary": msg.content[0].text})
```

### Use MediaPipe for better accuracy
```bash
pip install mediapipe
```
Then replace the Haar cascade detection in `app.py` with `mp.solutions.face_mesh` for 468-point landmark tracking.
