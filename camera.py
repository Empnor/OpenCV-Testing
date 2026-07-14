import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
import time
from ultralytics import YOLO

import database
import face_utils

MATCH_THRESHOLD = 0.45
MIN_DETECTION_CONFIDENCE = 0.8
MIN_EYE_DISTANCE_PIXELS = 30
PERSON_TRACKING_INTERVAL = 3


def has_reliable_eyes(face):
    """Quality check used before trusting a face detection for recognition -
    filters out low-confidence or too-small/angled faces (unreliable landmarks)."""
    confidence = face[14]
    if confidence < MIN_DETECTION_CONFIDENCE:
        return False
    right_eye = face[4:6]
    left_eye = face[6:8]
    eye_distance = np.linalg.norm(right_eye - left_eye)
    return eye_distance >= MIN_EYE_DISTANCE_PIXELS


def point_in_box(px, py, box):
    """Used to check whether a face's center point falls inside a tracked body's box -
    this is how a recognized name gets linked to the correct person."""
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def format_duration(seconds):
    """Formats a duration in seconds as '12s' or '1:23' for the on-screen timer."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def fit_to_size(image, target_w, target_h):
    """Scale image to fit within target dimensions, preserving aspect ratio,
    padding with black bars (letterbox) so nothing gets stretched/squished.
    Used only for display - detection runs on the original frame beforehand."""
    h, w = image.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return image

    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition — Camera")
        self.geometry("1000x750")

        database.init_db()

        # Video feed display - a plain Label widget that gets a new image
        # drawn into it every frame, which is what makes it look like video
        self.video_label = tk.Label(self, bg="black")
        self.video_label.pack(fill="both", expand=True)
        self.display_w, self.display_h = 960, 640 
        self.video_label.bind("<Configure>", self.on_resize) 

        # Manual refresh button - re-reads the database so newly-added
        # people (via admin_app.py) get picked up without restarting
        button_bar = tk.Frame(self)
        button_bar.pack(fill="x", pady=5)
        self.reload_btn = tk.Button(button_bar, text="Reload Known Faces", command=self.load_known_faces)
        self.reload_btn.pack(side="left", padx=5)

        self.status_label = tk.Label(button_bar, text="Loading models...")
        self.status_label.pack(side="left", padx=10)

        self.running = False
        self.cap = None

        # Load models once, up front (not per-frame) for performance
        self.face_detector = cv2.FaceDetectorYN.create(
            face_utils.DETECTOR_PATH, "", (640, 480),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        self.recognizer = face_utils.get_recognizer()
        self.person_model = YOLO("yolov8n.pt")  # pretrained body/person detector - no training required

        self.load_known_faces()
        self.protocol("WM_DELETE_WINDOW", self.on_close)  

        
        self.start()

    def on_resize(self, event):
        """Keeps track of the video label's current size so the displayed
        frame can be scaled to fit it (see fit_to_size)."""
        self.display_w, self.display_h = event.width, event.height

    def load_known_faces(self):
        """Re-reads the database - call this after adding people via the admin app."""
        known_faces = database.get_all_people_with_embeddings()
        self.names = list(known_faces.keys())
        self.embeddings = list(known_faces.values())
        self.status_label.config(text=f"{len(self.names)} known people loaded")

    def identify(self, face_embedding):
        """Compares a live face embedding against every known person in the
        database and returns the best match (or 'Unknown' if nothing clears
        the similarity threshold)."""
        best_name = "Unknown"
        best_score = -1.0
        for name, known_embedding in zip(self.names, self.embeddings):
            score = self.recognizer.match(face_embedding, known_embedding, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < MATCH_THRESHOLD:
            return "Unknown", best_score
        return best_name, best_score

    def start(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.running = True

        # Per-person tracking state, reset fresh each time the camera starts
        self.track_identities = {}   
        self.track_start_times = {}  
        self.track_last_seen = {}   
        self.last_person_boxes = {}  
        self.frame_count = 0

        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")

    def update_frame(self):
        """The main per-frame loop: detect bodies, detect+recognize faces,
        link them together, draw everything, and schedule the next frame.
        Runs continuously via Tkinter's .after() while self.running is True."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Could not read from webcam.")
            self.stop()
            return

        frame_h, frame_w = frame.shape[:2]
        self.frame_count += 1

        
        # YOLO's built-in tracker (persist=True) keeps the same ID for the
        # same person as they move around the frame, automatically.
        if self.frame_count % PERSON_TRACKING_INTERVAL == 0:
            person_results = self.person_model.track(
                frame, classes=[0], conf=0.4, persist=True, verbose=False
            )
            person_boxes = {}
            if person_results[0].boxes.id is not None:
                boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = person_results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, track_ids):
                    person_boxes[track_id] = tuple(box)
            self.last_person_boxes = person_boxes
        else:
            person_boxes = self.last_person_boxes

        # Update "how long has this person been in frame" bookkeeping
        now = time.time()
        for track_id in person_boxes:
            if track_id not in self.track_start_times:
                self.track_start_times[track_id] = now
            self.track_last_seen[track_id] = now

        self.face_detector.setInputSize((frame_w, frame_h))
        _, faces = self.face_detector.detect(frame)

        face_matches = []
        if faces is not None:
            for face in faces:
                fx, fy, fw, fh = face[:4].astype(int)
                if not has_reliable_eyes(face):
                    continue

                # Clamp the face box to stay within frame bounds - avoids a
                # broken crop for faces very close to the camera/frame edge
                clamped_face = face.copy()
                clamped_face[0] = max(0, clamped_face[0])
                clamped_face[1] = max(0, clamped_face[1])
                if clamped_face[0] + clamped_face[2] > frame_w:
                    clamped_face[2] = frame_w - clamped_face[0]
                if clamped_face[1] + clamped_face[3] > frame_h:
                    clamped_face[3] = frame_h - clamped_face[1]

                aligned = self.recognizer.alignCrop(frame, clamped_face)
                face_embedding = self.recognizer.feature(aligned)
                name, score = self.identify(face_embedding)

                if name == "Unknown":
                    continue

                center_x, center_y = fx + fw // 2, fy + fh // 2
                face_matches.append((center_x, center_y, name))

       
     
        for track_id in list(self.track_identities.keys()):
            if track_id not in person_boxes:
                continue
            self.track_identities[track_id]["frames_since_face"] += 1

     
        for center_x, center_y, name in face_matches:
            for track_id, box in person_boxes.items():
                if point_in_box(center_x, center_y, box):
                    self.track_identities[track_id] = {"name": name, "frames_since_face": 0}
                    break

      
        #   green  = name known, face currently visible
        #   orange = name known, but going off tracked body only (face not currently visible)
        #   gray   = person detected but never matched to a known face
        for track_id, box in person_boxes.items():
            x1, y1, x2, y2 = box

            if track_id in self.track_identities:
                name = self.track_identities[track_id]["name"]
                frames_since_face = self.track_identities[track_id]["frames_since_face"]
                if frames_since_face == 0:
                    label = f"{name} (face visible)"
                    color = (0, 255, 0)
                else:
                    label = f"{name} (tracked, ID {track_id})"
                    color = (0, 200, 255)
            else:
                label = f"Unknown person (ID {track_id})"
                color = (128, 128, 128)

            if track_id in self.track_start_times:
                duration = now - self.track_start_times[track_id]
                label += f" - {format_duration(duration)}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

       
        if faces is not None:
            for face in faces:
                fx, fy, fw, fh = face[:4].astype(int)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)

        
        display_frame = fit_to_size(frame, self.display_w, self.display_h)
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # keep a reference so it doesn't get garbage collected
        self.video_label.configure(image=imgtk)

        
        self.after(15, self.update_frame)

    def on_close(self):
        self.stop()
        self.destroy()


if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()