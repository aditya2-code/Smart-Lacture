

# 🎓 Smart-Lecture – Emotion Aware Video Player

### 🚀 Team: Nova Grid

An AI-powered intelligent video player that detects the learner’s facial expressions in real time and adapts the lecture experience accordingly. The system pauses videos when confusion or drowsiness is detected and assists the learner with summaries and alerts.

---

## 📌 Overview

Online learning platforms follow a static “one-size-fits-all” model where lecture videos continue playing regardless of the learner’s engagement level. This often leads to confusion, loss of focus, and poor knowledge retention.

**Smart-Lecture** solves this problem by creating an adaptive and interactive learning experience using Computer Vision and AI-driven emotion detection.

The system continuously monitors the learner through a webcam and reacts intelligently to improve understanding and attention.

---

## ✨ Features

* 🎭 Real-time facial expression detection
* 🤔 Detects learner confusion and pauses the lecture
* 📝 Generates a short summary of recent lecture content
* 😴 Detects drowsiness using eye movement analysis
* ⏸ Automatic pause with focus alert when drowsy
* 🎥 Interactive AI-powered video player

---

## 🧠 How It Works

1. The learner plays a lecture video in the smart player.
2. The webcam captures facial expressions in real time.
3. AI models analyze emotions continuously.
4. If confusion is detected → video pauses → summary displayed.
5. If drowsiness is detected → video pauses → alert shown.

This makes learning personalized, adaptive, and more engaging.

---

## 🛠 Tech Stack

### 👁️ Computer Vision & AI

* Python
* OpenCV
* Deep Learning CNN Model (Facial Expression Recognition)
* Dlib / Facial Landmark Detection

### 😴 Drowsiness Detection

* Eye Aspect Ratio (EAR)
* Real-time eye tracking algorithm

### 🎥 Video Player & Interface

* Python (Tkinter / PyQt) or Web-based UI
* HTML, CSS, JavaScript

### 🤖 NLP Summarization

* Transformer-based text summarization model
* NLP preprocessing techniques

---

## ⚙️ System Workflow

```
Lecture Video → Webcam Input → Face Detection →
Emotion Classification →
   ├── Confusion Detected → Pause Video → Show Summary
   └── Drowsiness Detected → Pause Video → Show Alert
```

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/smart-lecture.git
cd smart-lecture
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

---

## ▶️ Usage

1. Launch the application.
2. Load a lecture video.
3. Ensure webcam access is enabled.
4. Watch the lecture normally.
5. The system will automatically pause and assist based on detected emotions.

---

## 🚀 Future Scope

* AR/VR-based AI tutor avatars
* Voice-based doubt clarification
* Emotion-based adaptive playback speed
* Personalized learning analytics dashboard
* Classroom-level multi-student emotion monitoring
* Integration with LMS platforms (Moodle, Google Classroom)

---

## 🎯 Applications

* Smart e-learning platforms
* Online course providers
* Personalized AI tutors
* EdTech research & adaptive education systems

---

## 👥 Contributors

**Team Nova Grid**
AI & Full Stack Development Team

---

## 📜 License

This project is developed for academic and research purposes.
Open for educational use and future enhancements.

---

## 🙌 Acknowledgement

We aim to make online learning more interactive, intelligent, and student-focused by integrating AI-driven emotional awareness into digital education.

---
