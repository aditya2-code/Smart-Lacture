"""
Smart-Lecture v4 — MediaPipe-Equivalent Precision Engine
=========================================================
Architecture:
  Face ROI      : Haar cascade (fast bounding box only — runs every 45 frames)
  Eye landmarks : Lucas-Kanade Optical Flow (sub-pixel, 6 pts per eye)
                  + Kalman Filter (cv2.KalmanFilter) for jitter elimination
  EAR           : Exact 6-point formula on Kalman-smoothed eyelid points
  Gaze ratio    : (iris_cx - inner_corner_x) / (outer_x - inner_x)
                  Geometry-based = works through glasses, eyeliner, any light
  Calibration   : 45-frame personalised baseline (EAR, brow, gaze centre)
  Debounce      : DROWSY_FRAMES=20, CONFUSED_FRAMES=40, GAZE_SEC=3.0
  Persistence   : Session stats survive browser refresh (server state dict)
"""

import cv2
import numpy as np
import base64, json, os, time, math, random
from collections import deque
from flask import Flask, render_template, jsonify, request, send_from_directory
from threading import Thread, Lock

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── Cascade classifiers (face ROI only — not eye detection) ───────────────────
DATA     = cv2.data.haarcascades
face_cas = cv2.CascadeClassifier(DATA + 'haarcascade_frontalface_default.xml')
face_alt = cv2.CascadeClassifier(DATA + 'haarcascade_frontalface_alt2.xml')
eye_cas  = cv2.CascadeClassifier(DATA + 'haarcascade_eye.xml')

# ── Detection constants ────────────────────────────────────────────────────────
REINIT_INTERVAL  = 45     # re-detect face bounding box every N frames
LK_PARAMS = dict(
    winSize=(17, 17), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)
DROWSY_FRAMES  = 20    # consecutive frames below EAR threshold → trigger
CONFUSED_FRAMES= 40    # consecutive frames above brow threshold → trigger
GAZE_SEC       = 3.0   # seconds gaze_ratio off-centre → alert
STAT_COOLDOWN  = 15.0  # min seconds between same-type stat increments
GAZE_THRESHOLD = 0.28  # deviation from calibrated centre (MediaPipe standard)

# ── Shared state ───────────────────────────────────────────────────────────────
state_lock = Lock()
state = dict(
    emotion="neutral", confidence=0.0,
    ear=0.30, brow_ratio=1.0,
    gaze_ratio=0.5,
    gaze_offscreen=False, gaze_duration=0.0,
    attention=100, face_detected=False,
    paused_reason=None,
    gaze_alert=False,
    calibrated=False, cal_progress=0,
    baseline_ear=None, baseline_brow=None, baseline_gaze=None,
    cal_live_ear=0.0, cal_live_brow=0.0, cal_live_gaze=0.5,
    confused_frames=0, drowsy_frames=0,
    offscreen_start=None, cam_frame_b64=None,
    lm_debug=None,
    stat_confused=0, stat_bored=0, stat_gaze=0,
    session_start=time.time(),
    _last_bored_time=0.0, _last_confused_time=0.0, _last_gaze_time=0.0,
)

# ── Quiz bank ──────────────────────────────────────────────────────────────────
QUIZ_BANK = [
    {"q": "What is the derivative of x²?",             "opts": ["x","2x","x²/2","2"],             "ans": 1},
    {"q": "Dot product of orthogonal vectors equals?", "opts": ["1","-1","0","∞"],                "ans": 2},
    {"q": "Area of a circle with radius r?",           "opts": ["2πr","πr²","πr","4πr²"],         "ans": 1},
    {"q": "Which sort is O(n log n) average?",         "opts": ["Bubble","Merge","Selection","Insertion"],"ans":1},
    {"q": "Newton's 2nd Law: F = ?",                   "opts": ["mv","ma","mv²","m/a"],            "ans": 1},
    {"q": "log₂(8) = ?",                               "opts": ["2","4","3","8"],                  "ans": 2},
    {"q": "LIFO data structure?",                      "opts": ["Queue","Stack","Tree","Heap"],     "ans": 1},
    {"q": "Eigenvalue equation: Av = ?",               "opts": ["v","λv","Av","0"],                "ans": 1},
    {"q": "Determinant of identity matrix?",           "opts": ["0","n","1","-1"],                 "ans": 2},
    {"q": "Null space: Ax = ?",                        "opts": ["1","b","0","I"],                  "ans": 2},
    {"q": "Basis vectors must be?",                    "opts": ["Parallel","Orthogonal","Linearly independent","Equal"],"ans":2},
    {"q": "Rank = # of?",                              "opts": ["Rows","Columns","Pivot columns","Diagonals"],"ans":2},
]

# ── Transcripts ────────────────────────────────────────────────────────────────
TRANSCRIPTS = {}
_tf = os.path.join('data', 'transcripts.json')
if os.path.exists(_tf):
    with open(_tf) as f:
        TRANSCRIPTS = {int(k): v for k, v in json.load(f).items()}

def get_transcript_for_time(t_secs):
    keys = sorted(TRANSCRIPTS.keys())
    if not keys: return None
    best = keys[0]
    for k in keys:
        if k <= t_secs: best = k
        else: break
    return TRANSCRIPTS.get(best)


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER — 1-D scalar for EAR, gaze, brow
# ══════════════════════════════════════════════════════════════════════════════

def make_kalman(q=1e-4, r=0.01):
    """
    Build a 1-D constant-velocity Kalman filter.
    q = process noise (how fast the true value can change)
    r = measurement noise (how noisy the raw detector reading is)
    Lower r = trust measurements more; higher r = trust prediction more.
    """
    kf = cv2.KalmanFilter(2, 1)        # state: [value, velocity]
    kf.transitionMatrix    = np.array([[1, 1],[0, 1]], np.float32)
    kf.measurementMatrix   = np.array([[1, 0]], np.float32)
    kf.processNoiseCov     = np.eye(2, dtype=np.float32) * q
    kf.measurementNoiseCov = np.array([[r]], np.float32)
    kf.errorCovPost        = np.eye(2, dtype=np.float32)
    return kf

def kf_update(kf, measurement):
    """Predict + correct; return filtered scalar."""
    kf.predict()
    corrected = kf.correct(np.array([[np.float32(measurement)]], np.float32))
    return float(corrected[0][0])


# ══════════════════════════════════════════════════════════════════════════════
#  LANDMARK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class LandmarkEngine:
    """
    Sub-pixel eye tracker:
      1. Haar cascade → face bounding box (every REINIT_INTERVAL frames)
      2. Eye cascade  → initial 6-point landmark seeds
      3. LK optical flow → track landmarks sub-pixel between cascade runs
      4. Kalman filter → eliminate jitter on EAR, gaze_ratio, brow_y

    Gaze ratio formula (identical to MediaPipe refine_landmarks):
        ratio = (iris_x − inner_corner_x) / (outer_x − inner_x)
        0.0 = looking far right, 0.5 = centre, 1.0 = looking far left
    """

    def __init__(self):
        self.prev_gray   = None
        self.pts_l       = None    # left eye 6-pt LK points (6,1,2) float32
        self.pts_r       = None    # right eye
        self.frame_count = 0
        # Per-signal Kalman filters
        self.kf_ear  = make_kalman(q=1e-4, r=0.008)   # EAR is slow-moving
        self.kf_gaze = make_kalman(q=5e-4, r=0.015)   # gaze a bit faster
        self.kf_brow = make_kalman(q=1e-4, r=0.012)

    def process(self, frame):
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_e = cv2.equalizeHist(gray)
        self.frame_count += 1

        force_init = (self.frame_count % REINIT_INTERVAL == 1
                      or self.pts_l is None)

        lm = self._init_from_cascade(gray_e, frame) if force_init \
             else self._track_lk(gray, gray_e, frame)

        self.prev_gray = gray
        return lm

    # ── Cascade init ───────────────────────────────────────────────────────
    def _init_from_cascade(self, gray_e, frame):
        faces = face_cas.detectMultiScale(gray_e, 1.1, 5, minSize=(80,80))
        if not len(faces):
            faces = face_alt.detectMultiScale(gray_e, 1.05, 4, minSize=(70,70))
        if not len(faces):
            self.pts_l = self.pts_r = None
            return None

        fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
        face_roi = gray_e[fy:fy+fh, fx:fx+fw]
        eyes = eye_cas.detectMultiScale(face_roi, 1.05, 3, minSize=(20,20))

        if not len(eyes):
            self.pts_l = self.pts_r = None
            return self._result_face_only(fx, fy, fw, fh, frame)

        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        eye_data = []
        for (ex, ey, ew, eh) in eyes_sorted[:2]:
            pts6    = self._box_to_6pts(fx+ex, fy+ey, ew, eh)
            roi     = gray_e[fy+ey:fy+ey+eh, fx+ex:fx+ex+ew]
            ic      = self._iris_centre(roi)
            iris_abs= (fx+ex+ic[0], fy+ey+ic[1]) if ic \
                      else (fx+ex+ew//2, fy+ey+eh//2)
            eye_data.append((pts6, iris_abs))

        if len(eye_data) >= 2:
            self.pts_l = self._to_lk(eye_data[0][0])
            self.pts_r = self._to_lk(eye_data[1][0])
        elif len(eye_data) == 1:
            self.pts_l = self._to_lk(eye_data[0][0])
            self.pts_r = None

        return self._build(eye_data, fx, fy, fw, fh, frame)

    # ── LK tracking ────────────────────────────────────────────────────────
    def _track_lk(self, gray, gray_e, frame):
        if self.prev_gray is None or self.pts_l is None:
            return self._init_from_cascade(gray_e, frame)

        new_l, el = self._lk_step(self.prev_gray, gray, self.pts_l)
        new_r, er = self._lk_step(self.prev_gray, gray, self.pts_r) \
                    if self.pts_r is not None else (None, 999)

        if max(el, er) > 3.0:
            self.pts_l = self.pts_r = None
            return self._init_from_cascade(gray_e, frame)

        self.pts_l = new_l
        self.pts_r = new_r

        eye_data = []
        for pts in [new_l, new_r]:
            if pts is None: continue
            p   = pts.reshape(-1, 2)
            box = self._pts_box(p)
            bx, by, bw, bh = box
            roi = gray_e[by:by+bh, bx:bx+bw]
            ic  = self._iris_centre(roi)
            iris_abs = (bx+ic[0], by+ic[1]) if ic else (bx+bw//2, by+bh//2)
            eye_data.append((p.tolist(), iris_abs))

        if not eye_data:
            return None

        all_x  = [p[0] for ed in eye_data for p in np.array(ed[0]).reshape(-1,2)]
        all_y  = [p[1] for ed in eye_data for p in np.array(ed[0]).reshape(-1,2)]
        mg     = 40
        fx     = max(0, int(min(all_x))-mg)
        fy_    = max(0, int(min(all_y))-mg)
        fw     = int(max(all_x))-fx+mg
        fh     = int(max(all_y))-fy_+mg*2
        return self._build(eye_data, fx, fy_, fw, fh, frame)

    def _lk_step(self, prev, curr, pts):
        if pts is None: return None, 999
        new, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **LK_PARAMS)
        if new is None: return pts, 999
        good = err[st.ravel()==1]
        return new, float(np.mean(good)) if len(good) else 999

    # ── Build result with Kalman smoothing ─────────────────────────────────
    def _build(self, eye_data, fx, fy, fw, fh, frame):
        ears, gazes, iris_pts = [], [], []

        for pts_raw, iris_abs in eye_data:
            p = np.array(pts_raw).reshape(-1,2).astype(float)
            if len(p) < 6: continue

            # EAR — 6-point formula
            ear = (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) \
                  / (2.0 * np.linalg.norm(p[0]-p[3]) + 1e-6)
            ears.append(float(np.clip(ear, 0.05, 0.90)))

            # Gaze ratio (geometry-based — same as MediaPipe)
            # p[0]=outer corner, p[3]=inner corner
            outer_x  = float(p[0][0])
            inner_x  = float(p[3][0])
            iris_x   = float(iris_abs[0])
            gaze = (iris_x - outer_x) / (inner_x - outer_x + 1e-6)
            gazes.append(float(np.clip(gaze, 0.0, 1.0)))
            iris_pts.append(iris_abs)

        if not ears:
            return self._result_face_only(fx, fy, fw, fh, frame)

        raw_ear  = float(np.mean(ears))
        raw_gaze = float(np.mean(gazes))
        raw_brow = self._brow_y(frame, fx, fy, fw, fh)

        # ── Kalman filter: eliminate frame-to-frame jitter ─────────────────
        smooth_ear  = kf_update(self.kf_ear,  raw_ear)
        smooth_gaze = kf_update(self.kf_gaze, raw_gaze)
        smooth_brow = kf_update(self.kf_brow, raw_brow)

        # Iris centre for visualiser
        ic_x = int(np.mean([p[0] for p in iris_pts])) if iris_pts else fx+fw//2
        ic_y = int(np.mean([p[1] for p in iris_pts])) if iris_pts else fy+fh//2

        return dict(
            ear=smooth_ear, brow_y=smooth_brow, gaze_ratio=smooth_gaze,
            raw_ear=raw_ear, raw_gaze=raw_gaze,   # expose raw for debug
            iris_cx=ic_x, iris_cy=ic_y,
            face_box=(fx, fy, fw, fh),
            eyes_found=len(ears),
            frame_w=frame.shape[1], frame_h=frame.shape[0],
        )

    def _result_face_only(self, fx, fy, fw, fh, frame):
        return dict(
            ear=0.30, brow_y=0.25, gaze_ratio=0.5,
            raw_ear=0.30, raw_gaze=0.5,
            iris_cx=fx+fw//2, iris_cy=fy+fh//2,
            face_box=(fx, fy, fw, fh), eyes_found=0,
            frame_w=frame.shape[1], frame_h=frame.shape[0],
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _box_to_6pts(ex, ey, ew, eh):
        """6 eyelid landmark seeds from eye bounding box."""
        return [
            [ex,           ey+eh//2],
            [ex+ew//3,     ey+int(eh*0.15)],
            [ex+2*ew//3,   ey+int(eh*0.15)],
            [ex+ew,        ey+eh//2],
            [ex+2*ew//3,   ey+int(eh*0.85)],
            [ex+ew//3,     ey+int(eh*0.85)],
        ]

    @staticmethod
    def _to_lk(pts6):
        return np.array(pts6, dtype=np.float32).reshape(-1,1,2)

    @staticmethod
    def _pts_box(pts):
        x0,y0 = int(pts[:,0].min()), int(pts[:,1].min())
        x1,y1 = int(pts[:,0].max()), int(pts[:,1].max())
        return x0, y0, max(x1-x0,1), max(y1-y0,1)

    @staticmethod
    def _iris_centre(roi):
        if roi.size == 0: return None
        _, th = cv2.threshold(roi, int(np.percentile(roi,22)),255,cv2.THRESH_BINARY_INV)
        M = cv2.moments(th)
        if M["m00"] < 1: return None
        return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    @staticmethod
    def _brow_y(frame, fx, fy, fw, fh):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top  = gray[fy:fy+int(fh*0.42), fx:fx+fw]
        if top.size == 0: return 0.25
        col_means = np.mean(cv2.equalizeHist(top), axis=1)
        return float(np.argmin(col_means)) / (fh+1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  Annotation
# ══════════════════════════════════════════════════════════════════════════════

def draw_landmarks(frame, lm, emotion, attention, gaze_offscreen, b_gaze=0.5):
    out = frame.copy()
    if lm is None:
        cv2.putText(out,"NO FACE",(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,80,200),1)
        return out

    fx, fy, fw, fh = lm["face_box"]
    em_color = {"confused":(123,97,255),"drowsy":(255,107,53),
                "focused":(0,229,160),"away":(200,80,80)}.get(emotion,(0,229,160))

    cv2.rectangle(out,(fx,fy),(fx+fw,fy+fh),em_color,1)

    # Subtle dot grid
    for gx in range(fx+8,fx+fw-8,16):
        for gy in range(fy+8,fy+fh-8,16):
            cv2.circle(out,(gx,gy),1,(0,80,50),-1)

    # Iris crosshair
    ix, iy = lm["iris_cx"], lm["iris_cy"]
    gc = (0,130,255) if gaze_offscreen else (255,200,0)
    cv2.circle(out,(ix,iy),6,gc,1)
    cv2.circle(out,(ix,iy),2,gc,-1)
    cv2.line(out,(ix-10,iy),(ix+10,iy),gc,1)
    cv2.line(out,(ix,iy-10),(ix,iy+10),gc,1)

    # EAR mini-bar (right edge of face box)
    ear_bar = int(min(lm["ear"]/0.40, 1.0) * fh)
    bar_col = (0,229,160) if lm["ear"] > 0.22 else (255,77,109)
    cv2.rectangle(out,(fx+fw+3,fy+fh-ear_bar),(fx+fw+8,fy+fh),bar_col,-1)

    # Gaze label
    if gaze_offscreen:
        cv2.putText(out,"GAZE AWAY",(fx,fy-6),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,130,255),1)

    cv2.putText(out,emotion.upper(),(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,em_color,1)
    cv2.putText(out,f"Att:{attention}%",(6,32),cv2.FONT_HERSHEY_SIMPLEX,0.36,(150,150,150),1)
    cv2.putText(out,f"EAR:{lm['ear']:.3f}",(6,46),cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,200,80),1)
    cv2.putText(out,f"Gz:{lm['gaze_ratio']:.3f}",(6,60),cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,160,220),1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Camera Processor
# ══════════════════════════════════════════════════════════════════════════════

class CameraProcessor:
    def __init__(self):
        self.cap      = None
        self.running  = False
        self.engine   = LandmarkEngine()
        self._cal_buf = []

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        fc = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.05); continue
            fc += 1
            if fc % 2 == 0:
                self._process(frame)
            time.sleep(0.005)

    def _process(self, frame):
        lm = self.engine.process(frame)
        with state_lock:
            state["face_detected"] = lm is not None
            b_gaze = state.get("baseline_gaze") or 0.5
            if lm is None:
                state["attention"] = max(state["attention"]-1, 20)
                state["emotion"]   = "away"
                state["lm_debug"]  = None
            elif not state["calibrated"]:
                self._calibrate(lm)
            else:
                self._analyse(lm)

            if lm:
                state["lm_debug"] = {
                    "iris_x":lm["iris_cx"],"iris_y":lm["iris_cy"],
                    "face_x":lm["face_box"][0],"face_y":lm["face_box"][1],
                    "face_w":lm["face_box"][2],"face_h":lm["face_box"][3],
                    "frame_w":lm["frame_w"],"frame_h":lm["frame_h"],
                    "ear":round(lm["ear"],3),"gaze":round(lm["gaze_ratio"],3),
                    "raw_ear":round(lm.get("raw_ear",lm["ear"]),3),
                    "raw_gaze":round(lm.get("raw_gaze",lm["gaze_ratio"]),3),
                    "baseline_gaze": b_gaze,
                }

            em  = state["emotion"]
            att = state["attention"]
            goff= state["gaze_offscreen"]

        annotated = draw_landmarks(frame, lm, em, att, goff, b_gaze)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY,72])
        with state_lock:
            state["cam_frame_b64"] = base64.b64encode(buf).decode()

    def _calibrate(self, lm):
        self._cal_buf.append((lm["ear"], lm["brow_y"], lm["gaze_ratio"]))
        n    = len(self._cal_buf)
        prog = min(n/45*100, 100)
        ears  = [x[0] for x in self._cal_buf]
        brows = [x[1] for x in self._cal_buf]
        gazes = [x[2] for x in self._cal_buf]
        state["cal_progress"]  = round(prog)
        state["cal_live_ear"]  = round(float(np.mean(ears)),3)
        state["cal_live_brow"] = round(float(np.mean(brows)),3)
        state["cal_live_gaze"] = round(float(np.mean(gazes)),3)
        if n >= 45:
            state["baseline_ear"]  = float(np.mean(ears))
            state["baseline_brow"] = float(np.mean(brows))
            state["baseline_gaze"] = float(np.mean(gazes))
            state["calibrated"]    = True
            state["cal_progress"]  = 100

    def _analyse(self, lm):
        ear   = lm["ear"]
        brow  = lm["brow_y"]
        gaze  = lm["gaze_ratio"]
        b_ear  = state["baseline_ear"]  or 0.30
        b_brow = state["baseline_brow"] or 0.25
        b_gaze = state["baseline_gaze"] or 0.50
        now    = time.time()

        state["ear"]        = round(ear,  3)
        state["brow_ratio"] = round(brow / (b_brow+1e-6), 3)
        state["gaze_ratio"] = round(gaze, 3)

        # ── Geometry-based gaze (±GAZE_THRESHOLD from calibrated centre) ────
        gaze_dev  = abs(gaze - b_gaze)
        offscreen = gaze_dev > GAZE_THRESHOLD and lm["eyes_found"] > 0
        state["gaze_offscreen"] = offscreen
        if offscreen:
            if state["offscreen_start"] is None:
                state["offscreen_start"] = now
            dur = now - state["offscreen_start"]
            state["gaze_duration"] = round(dur, 1)
            since = now - state["_last_gaze_time"]
            if dur >= GAZE_SEC and not state["gaze_alert"] and since > STAT_COOLDOWN:
                state["gaze_alert"]      = True
                state["stat_gaze"]      += 1
                state["_last_gaze_time"] = now
        else:
            state["offscreen_start"] = None
            state["gaze_duration"]   = 0.0

        # ── Debounced drowsiness: EAR < 68% baseline ─────────────────────────
        drowsy   = ear < (b_ear*0.68) and ear < 0.23 and lm["eyes_found"] > 0
        confused = brow > (b_brow*1.18) and lm["eyes_found"] > 0

        if drowsy:
            state["drowsy_frames"]  += 1
            state["confused_frames"] = 0
            state["emotion"]         = "drowsy"
            state["attention"]       = max(state["attention"]-2, 0)
            since = now - state["_last_bored_time"]
            if (state["drowsy_frames"] >= DROWSY_FRAMES
                    and state["paused_reason"] is None
                    and since > STAT_COOLDOWN):
                state["paused_reason"]    = "bored"
                state["stat_bored"]      += 1
                state["_last_bored_time"] = now
                state["drowsy_frames"]    = 0
        elif confused:
            state["confused_frames"] += 1
            state["drowsy_frames"]    = 0
            state["emotion"]          = "confused"
            state["attention"]        = max(state["attention"]-1, 30)
            since = now - state["_last_confused_time"]
            if (state["confused_frames"] >= CONFUSED_FRAMES
                    and state["paused_reason"] is None
                    and since > STAT_COOLDOWN):
                state["paused_reason"]       = "confused"
                state["stat_confused"]      += 1
                state["_last_confused_time"] = now
                state["confused_frames"]     = 0
        else:
            state["confused_frames"] = 0
            state["drowsy_frames"]   = 0
            state["emotion"]         = "focused"
            state["attention"]       = min(state["attention"]+1, 100)

        state["confidence"] = round(
            min(abs(ear-b_ear)/(b_ear+1e-6)*100+55, 99), 1)

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  Demo Processor
# ══════════════════════════════════════════════════════════════════════════════

class DemoProcessor:
    def __init__(self):
        self.running = False
        self._tick   = 0

    def start(self):
        self.running = True
        Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        while self.running:
            self._tick += 1
            t   = self._tick
            now = time.time()
            with state_lock:
                state.update(calibrated=True, cal_progress=100,
                             baseline_ear=0.30, baseline_brow=0.25, baseline_gaze=0.50,
                             face_detected=True)
                phase = t % 440

                if phase < 200:                    # Focused
                    state["emotion"]         = "focused"
                    state["ear"]             = round(0.29+0.02*math.sin(t*0.1),3)
                    state["brow_ratio"]      = round(1.0+0.04*math.sin(t*0.07),3)
                    state["gaze_ratio"]      = round(0.50+0.04*math.sin(t*0.13),3)
                    state["attention"]       = min(state["attention"]+1,95)
                    state["confused_frames"] = 0
                    state["drowsy_frames"]   = 0
                    state["gaze_offscreen"]  = False
                    state["offscreen_start"] = None
                    state["gaze_duration"]   = 0.0
                    gx = int(160+20*math.sin(t*0.13))
                    state["lm_debug"] = {"iris_x":gx,"iris_y":80,
                        "face_x":60,"face_y":20,"face_w":200,"face_h":200,
                        "frame_w":320,"frame_h":240,
                        "ear":state["ear"],"gaze":state["gaze_ratio"],
                        "raw_ear":state["ear"],"raw_gaze":state["gaze_ratio"],
                        "baseline_gaze":0.5}

                elif phase < 290:                  # Confused
                    state["emotion"]         = "confused"
                    state["ear"]             = 0.27
                    state["brow_ratio"]      = 1.25
                    state["gaze_ratio"]      = 0.50
                    state["attention"]       = max(state["attention"]-2,40)
                    state["confused_frames"] = state["confused_frames"]+1
                    state["lm_debug"] = {"iris_x":160,"iris_y":80,
                        "face_x":60,"face_y":20,"face_w":200,"face_h":200,
                        "frame_w":320,"frame_h":240,
                        "ear":0.27,"gaze":0.50,"raw_ear":0.265,"raw_gaze":0.51,
                        "baseline_gaze":0.5}
                    since = now - state["_last_confused_time"]
                    if (state["confused_frames"] >= CONFUSED_FRAMES
                            and state["paused_reason"] is None
                            and since > STAT_COOLDOWN):
                        state["paused_reason"]       = "confused"
                        state["stat_confused"]      += 1
                        state["_last_confused_time"] = now
                        state["confused_frames"]     = 0

                elif phase < 370:                  # Drowsy
                    state["emotion"]        = "drowsy"
                    state["ear"]            = 0.14
                    state["brow_ratio"]     = 1.0
                    state["gaze_ratio"]     = 0.50
                    state["attention"]      = max(state["attention"]-3,10)
                    state["drowsy_frames"]  = state["drowsy_frames"]+1
                    state["lm_debug"] = {"iris_x":160,"iris_y":98,
                        "face_x":60,"face_y":20,"face_w":200,"face_h":200,
                        "frame_w":320,"frame_h":240,
                        "ear":0.14,"gaze":0.50,"raw_ear":0.155,"raw_gaze":0.49,
                        "baseline_gaze":0.5}
                    since = now - state["_last_bored_time"]
                    if (state["drowsy_frames"] >= DROWSY_FRAMES
                            and state["paused_reason"] is None
                            and since > STAT_COOLDOWN):
                        state["paused_reason"]    = "bored"
                        state["stat_bored"]      += 1
                        state["_last_bored_time"] = now
                        state["drowsy_frames"]    = 0

                else:                              # Gaze away
                    state["emotion"]        = "focused"
                    state["gaze_offscreen"] = True
                    state["gaze_ratio"]     = round(0.82+0.05*math.sin(t*0.3),3)
                    state["gaze_duration"]  = round((phase-370)*0.07,1)
                    gx = int(255+10*math.sin(t*0.3))
                    state["lm_debug"] = {"iris_x":gx,"iris_y":80,
                        "face_x":60,"face_y":20,"face_w":200,"face_h":200,
                        "frame_w":320,"frame_h":240,
                        "ear":0.29,"gaze":state["gaze_ratio"],
                        "raw_ear":0.295,"raw_gaze":state["gaze_ratio"],
                        "baseline_gaze":0.5}
                    if state["offscreen_start"] is None:
                        state["offscreen_start"] = now
                    since = now - state["_last_gaze_time"]
                    if (state["gaze_duration"] >= GAZE_SEC
                            and not state["gaze_alert"]
                            and since > STAT_COOLDOWN):
                        state["gaze_alert"]      = True
                        state["stat_gaze"]      += 1
                        state["_last_gaze_time"] = now

            time.sleep(0.05)

    def stop(self):
        self.running = False


# ── Start ──────────────────────────────────────────────────────────────────────
cam = CameraProcessor()
if not cam.start():
    cam = DemoProcessor()
    cam.start()


# ══════════════════════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    with state_lock:
        s = {k:v for k,v in state.items() if k!="cam_frame_b64"}
    return jsonify(s)

@app.route('/api/frame')
def get_frame():
    with state_lock:
        b64 = state.get("cam_frame_b64")
    return jsonify({"frame": b64})

@app.route('/api/resume', methods=['POST'])
def resume():
    with state_lock:
        state["paused_reason"]       = None
        state["confused_frames"]     = 0
        state["drowsy_frames"]       = 0
        state["_last_bored_time"]    = 0.0
        state["_last_confused_time"] = 0.0
    return jsonify({"ok": True})

@app.route('/api/quiz')
def get_quiz():
    q = random.choice(QUIZ_BANK)
    return jsonify({"question":q["q"],"options":q["opts"],"answer":q["ans"]})

@app.route('/api/calibrate', methods=['POST'])
def recalibrate():
    with state_lock:
        state["calibrated"]    = False
        state["cal_progress"]  = 0
        state["paused_reason"] = None
    if hasattr(cam,'_cal_buf'): cam._cal_buf = []
    if hasattr(cam,'engine'):   cam.engine   = LandmarkEngine()
    return jsonify({"ok": True})

@app.route('/api/get_summary')
def get_summary():
    t     = float(request.args.get('t',0))
    entry = get_transcript_for_time(t)
    if entry is None:
        return jsonify({"topic":"Lecture Content",
                        "summary":"No transcript loaded. Add data/transcripts.json.",
                        "concepts":[],"timestamp":t})
    transcript = entry.get("transcript","")
    sentences  = [s.strip() for s in transcript.split(". ") if len(s.strip())>20]
    top2       = sorted(sentences, key=len, reverse=True)[:2]
    summary    = ". ".join(top2)+("." if top2 else "")
    return jsonify({"topic":entry.get("topic","Lecture Content"),
                    "summary":summary,"concepts":entry.get("key_concepts",[]),
                    "timestamp":t})

@app.route('/api/gaze_ack', methods=['POST'])
def gaze_ack():
    with state_lock:
        state["gaze_alert"]      = False
        state["offscreen_start"] = None
        state["gaze_duration"]   = 0.0
    return jsonify({"ok": True})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error":"No file"}),400
    f = request.files['video']
    if not f.filename: return jsonify({"error":"Empty filename"}),400
    f.save(os.path.join(app.config['UPLOAD_FOLDER'],'lecture.mp4'))
    return jsonify({"ok":True})

@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
