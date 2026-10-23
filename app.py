import time
import math
import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Drowsiness Detector", page_icon="😴", layout="wide")
st.title("😴 Demo 3: Facial Analysis & Drowsiness Detector")

EAR_CLOSED_THRESH = 0.21
EYE_CLOSED_FRAMES_ALERT = 28
MAR_YAWN_THRESH = 0.60
YAWN_FRAMES_ALERT = 20

RIGHT_EYE_PTS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_PTS = [362, 385, 387, 263, 373, 380]
MOUTH_PTS = [61, 291, 13, 14, 82, 312, 87, 317]
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

def euclidean_dist(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def calculate_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices]
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    h_dist = euclidean_dist(pts[0], pts[3])
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)

def calculate_mar(landmarks, mouth_indices, w, h):
    p_corner_l = (int(landmarks[mouth_indices[0]].x * w), int(landmarks[mouth_indices[0]].y * h))
    p_corner_r = (int(landmarks[mouth_indices[1]].x * w), int(landmarks[mouth_indices[1]].y * h))
    p_lip_top = (int(landmarks[mouth_indices[2]].x * w), int(landmarks[mouth_indices[2]].y * h))
    p_lip_bot = (int(landmarks[mouth_indices[3]].x * w), int(landmarks[mouth_indices[3]].y * h))

    vertical = euclidean_dist(p_lip_top, p_lip_bot)
    horizontal = euclidean_dist(p_corner_l, p_corner_r)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def estimate_gaze(landmarks, w, h):
    r_outer = landmarks[33].x * w
    r_inner = landmarks[133].x * w
    r_iris = landmarks[RIGHT_IRIS_CENTER].x * w

    l_inner = landmarks[362].x * w
    l_outer = landmarks[263].x * w
    l_iris = landmarks[LEFT_IRIS_CENTER].x * w

    r_ratio = (r_iris - min(r_outer, r_inner)) / max(abs(r_inner - r_outer), 1e-5)
    l_ratio = (l_iris - min(l_inner, l_outer)) / max(abs(l_outer - l_inner), 1e-5)
    avg_ratio = (r_ratio + l_ratio) / 2.0

    if avg_ratio < 0.38:
        return "LOOKING RIGHT"
    elif avg_ratio > 0.62:
        return "LOOKING LEFT"
    else:
        return "LOOKING CENTER"

def draw_glass_card(img, x, y, w, h, bg_color=(20, 24, 33), alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 80, 100), 1, cv2.LINE_AA)

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.closed_eye_frame_count = 0
        self.yawn_frame_count = 0
        self.total_blinks = 0
        self.eye_was_closed = False
        self.show_mesh = False
        self.prev_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.flip(img, 1)
        h, w, _ = frame_bgr.shape

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-5)
        self.prev_time = current_time

        drowsy_alert = False
        yawn_alert = False
        gaze_str = "DETECTING..."
        avg_ear = 0.0
        mar = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            r_ear = calculate_ear(landmarks, RIGHT_EYE_PTS, w, h)
            l_ear = calculate_ear(landmarks, LEFT_EYE_PTS, w, h)
            avg_ear = (r_ear + l_ear) / 2.0

            mar = calculate_mar(landmarks, MOUTH_PTS, w, h)
            gaze_str = estimate_gaze(landmarks, w, h)

            if avg_ear < EAR_CLOSED_THRESH:
                self.closed_eye_frame_count += 1
                self.eye_was_closed = True
                if self.closed_eye_frame_count >= EYE_CLOSED_FRAMES_ALERT:
                    drowsy_alert = True
            else:
                if self.eye_was_closed and self.closed_eye_frame_count < EYE_CLOSED_FRAMES_ALERT:
                    self.total_blinks += 1
                self.closed_eye_frame_count = 0
                self.eye_was_closed = False

            if mar > MAR_YAWN_THRESH:
                self.yawn_frame_count += 1
                if self.yawn_frame_count >= YAWN_FRAMES_ALERT:
                    yawn_alert = True
            else:
                self.yawn_frame_count = max(0, self.yawn_frame_count - 1)

            if self.show_mesh:
                self.mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

            r_iris_pt = (int(landmarks[RIGHT_IRIS_CENTER].x * w), int(landmarks[RIGHT_IRIS_CENTER].y * h))
            l_iris_pt = (int(landmarks[LEFT_IRIS_CENTER].x * w), int(landmarks[LEFT_IRIS_CENTER].y * h))
            cv2.circle(frame_bgr, r_iris_pt, 4, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, l_iris_pt, 4, (255, 255, 0), -1, cv2.LINE_AA)

        draw_glass_card(frame_bgr, 20, 20, 360, 210, bg_color=(15, 20, 28), alpha=0.82)
        cv2.putText(frame_bgr, "AI DRIVER & FATIGUE MONITOR", (36, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

        if face_detected:
            eye_status = "CLOSED" if (avg_ear < EAR_CLOSED_THRESH) else "OPEN"
            eye_color = (0, 70, 255) if eye_status == "CLOSED" else (0, 255, 120)
            cv2.putText(frame_bgr, f"Eyes: {eye_status} (EAR: {avg_ear:.2f})", (36, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, eye_color, 2, cv2.LINE_AA)

            cv2.putText(frame_bgr, f"Gaze: {gaze_str}", (36, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

            yawn_status = "YAWNING" if yawn_alert else "NORMAL"
            yawn_color = (0, 140, 255) if yawn_alert else (200, 200, 200)
            cv2.putText(frame_bgr, f"Mouth: {yawn_status} (MAR: {mar:.2f})", (36, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, yawn_color, 1, cv2.LINE_AA)

            cv2.putText(frame_bgr, f"Blinks: {self.total_blinks}   FPS: {fps:.0f}", (36, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            fatigue_score = min(100, int((self.closed_eye_frame_count / EYE_CLOSED_FRAMES_ALERT) * 80 + (self.yawn_frame_count / YAWN_FRAMES_ALERT) * 40))
            bar_w = 320
            fill_w = int(bar_w * (fatigue_score / 100.0))
            bar_color = (0, 255, 120) if fatigue_score < 40 else ((0, 180, 255) if fatigue_score < 75 else (0, 40, 255))
            cv2.rectangle(frame_bgr, (36, 192), (36 + bar_w, 206), (50, 50, 50), -1)
            if fill_w > 0:
                cv2.rectangle(frame_bgr, (36, 192), (36 + fill_w, 206), bar_color, -1)
            cv2.putText(frame_bgr, f"Fatigue: {fatigue_score}%", (40, 204),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "No Face Detected", (36, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        if drowsy_alert:
            pulse = int((math.sin(time.time() * 12) + 1) * 127)
            cv2.rectangle(frame_bgr, (0, 0), (w, h), (0, 0, pulse), 10)
            cw, ch = 680, 90
            cx, cy = (w - cw) // 2, h - 140
            draw_glass_card(frame_bgr, cx, cy, cw, ch, bg_color=(0, 0, 160), alpha=0.88)
            cv2.putText(frame_bgr, "WARNING: SLEEPY / DROWSY DRIVER!", (cx + 25, cy + 58),
                        cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

        elif yawn_alert:
            cw, ch = 520, 70
            cx, cy = (w - cw) // 2, h - 120
            draw_glass_card(frame_bgr, cx, cy, cw, ch, bg_color=(0, 90, 180), alpha=0.85)
            cv2.putText(frame_bgr, "FATIGUE ALERT: YAWNING DETECTED", (cx + 25, cy + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

show_mesh_toggle = st.sidebar.checkbox("Show 478-Point Facial Mesh", value=False)

ctx = webrtc_streamer(
    key="drowsiness-demo",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if ctx.video_processor:
    ctx.video_processor.show_mesh = show_mesh_toggle