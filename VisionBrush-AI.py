"""
AI Hand Painter
----------------------------------------
A hand‑tracking drawing program using OpenCV + MediaPipe.
Tracks your index finger through the live camera and lets you draw on screen.

Install dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python hand_paint.py

Gesture Guide:
    - Only index finger up  -> Free drawing / moving (Draw / Move)
    - Pinch (thumb + index) -> "Click" to select tools/colors from the toolbar,
                               or to set start/end points for shapes (line, rectangle, circle)
    - Closed fist (no fingers up) -> Pause / no action

Keyboard shortcuts (optional):
    q -> Quit
    s -> Save PNG snapshot
    r -> Start/stop video recording
    c -> Clear canvas
    +/- -> Increase/decrease brush thickness
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
from datetime import datetime

# ----------------------------------------------------------------------
# 1) Ask user for dominant hand (right-handed or left-handed)
# ----------------------------------------------------------------------
def ask_handedness():
    while True:
        ans = input("Are you right-handed or left-handed? (r = right / l = left): ").strip().lower()
        if ans in ("r", "right"):
            return "Right"
        if ans in ("l", "left"):
            return "Left"
        print("Please enter only r or l.")

DOMINANT_HAND = ask_handedness()
print(f"Okay! The toolbar will appear on the {'right' if DOMINANT_HAND == 'Right' else 'left'} side.")
print("Opening camera... please wait.")

# ----------------------------------------------------------------------
# 2) General settings
# ----------------------------------------------------------------------
CAM_W, CAM_H = 1280, 720
TOOLBAR_W = 130          # smaller toolbar width
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 50
PINCH_THRESHOLD = 40
CLICK_COOLDOWN = 0.35

OUTPUT_DIR = os.path.join(os.getcwd(), "hand_paint_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [
    ("Red",   (0, 0, 255)),
    ("Green", (0, 200, 0)),
    ("Blue",  (255, 0, 0)),
    ("Yellow",(0, 220, 220)),
    ("Purple",(200, 0, 200)),
    ("White", (255, 255, 255)),
    ("Black", (0, 0, 0)),
]

SHAPES = ["Free", "Line", "Rectangle", "Circle", "Text", "Eraser"]

# ----------------------------------------------------------------------
# 3) MediaPipe Hands setup
# ----------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ----------------------------------------------------------------------
# 4) Program state
# ----------------------------------------------------------------------
class State:
    def __init__(self):
        self.current_color = COLORS[0][1]
        self.current_shape = "Free"
        self.brush_thickness = BRUSH_THICKNESS
        self.prev_point = None
        self.shape_start = None
        self.last_click_time = 0
        self.recording = False
        self.video_writer = None
        self.status_msg = ""
        self.status_msg_time = 0

state = State()

canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

# ----------------------------------------------------------------------
# 5) Build toolbar layout (smaller buttons)
# ----------------------------------------------------------------------
def build_toolbar_buttons():
    """Returns button rectangles: (x1, y1, x2, y2, kind, value)"""
    buttons = []
    if DOMINANT_HAND == "Right":
        x1, x2 = CAM_W - TOOLBAR_W, CAM_W
    else:
        x1, x2 = 0, TOOLBAR_W

    y = 15
    button_height = 35
    button_spacing = 40

    # Color buttons
    for name, color in COLORS:
        buttons.append((x1 + 10, y, x2 - 10, y + button_height, "color", color))
        y += button_spacing

    y += 5
    # Shape/tool buttons
    for shape in SHAPES:
        buttons.append((x1 + 10, y, x2 - 10, y + button_height, "shape", shape))
        y += button_spacing

    y += 5
    # Action buttons
    buttons.append((x1 + 10, y, x2 - 10, y + button_height, "action", "clear"))
    y += button_spacing
    buttons.append((x1 + 10, y, x2 - 10, y + button_height, "action", "save"))
    y += button_spacing
    buttons.append((x1 + 10, y, x2 - 10, y + button_height, "action", "record"))
    y += button_spacing

    return buttons, (x1, x2)

TOOLBAR_BUTTONS, (TOOLBAR_X1, TOOLBAR_X2) = build_toolbar_buttons()

# ----------------------------------------------------------------------
# 6) Gesture helper functions
# ----------------------------------------------------------------------
def fingers_up(landmarks, handedness_label):
    """Returns [thumb, index, middle, ring, pinky] with 1=up, 0=down."""
    fingers = []

    # Thumb: compare x depending on hand
    if handedness_label == "Right":
        fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)
    else:
        fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

    return fingers

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def point_in_button(x, y, button):
    x1, y1, x2, y2, _, _ = button
    return x1 <= x <= x2 and y1 <= y <= y2

def set_status(msg):
    state.status_msg = msg
    state.status_msg_time = time.time()

# ----------------------------------------------------------------------
# 7) Draw toolbar on frame (smaller text)
# ----------------------------------------------------------------------
def draw_toolbar(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (TOOLBAR_X1, 0), (TOOLBAR_X2, CAM_H), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    for (x1, y1, x2, y2, kind, value) in TOOLBAR_BUTTONS:
        if kind == "color":
            cv2.rectangle(frame, (x1, y1), (x2, y2), value, -1)
            border = (255, 255, 255) if value == state.current_color else (90, 90, 90)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
        elif kind == "shape":
            selected = value == state.current_shape
            bg = (70, 70, 70) if not selected else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
            cv2.putText(frame, value, (x1 + 6, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        elif kind == "action":
            label = {
                "clear": "Clear",
                "save": "Save PNG",
                "record": ("Stop Recording" if state.recording else "Record Video")
            }[value]
            bg = (0, 0, 160) if value == "record" and state.recording else (60, 60, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
            cv2.putText(frame, label, (x1 + 6, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame

# ----------------------------------------------------------------------
# 8) Button actions
# ----------------------------------------------------------------------
def activate_button(button):
    kind, value = button[4], button[5]
    if kind == "color":
        state.current_color = value
        set_status("Color selected")
    elif kind == "shape":
        state.current_shape = value
        state.shape_start = None
        set_status(f"Tool: {value}")
    elif kind == "action":
        if value == "clear":
            canvas[:] = 0
            set_status("Canvas cleared")
        elif value == "save":
            save_image()
        elif value == "record":
            toggle_recording()

def save_image():
    fname = os.path.join(OUTPUT_DIR, f"painting_{datetime.now():%Y%m%d_%H%M%S}.png")
    cv2.imwrite(fname, canvas)
    set_status(f"Saved: {os.path.basename(fname)}")

def toggle_recording():
    if not state.recording:
        fname = os.path.join(OUTPUT_DIR, f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        state.video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (CAM_W, CAM_H))
        state.recording = True
        set_status("Recording started")
    else:
        state.recording = False
        if state.video_writer is not None:
            state.video_writer.release()
            state.video_writer = None
        set_status("Recording stopped")

# ----------------------------------------------------------------------
# 9) Main loop
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("Camera not found! Please check your webcam connection.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        index_tip = None
        thumb_tip = None
        is_pinch = False
        fing = [0, 0, 0, 0, 0]

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            handed_label = DOMINANT_HAND
            if result.multi_handedness:
                handed_label = result.multi_handedness[0].classification[0].label

            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * CAM_W), int(lm.y * CAM_H)))

            fing = fingers_up(lm_list, handed_label)
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]
            is_pinch = distance(index_tip, thumb_tip) < PINCH_THRESHOLD

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        now = time.time()
        only_index_up = fing[1] == 1 and fing[2] == 0 and fing[3] == 0 and fing[4] == 0

        # Interaction logic
        if index_tip is not None:
            x, y = index_tip
            in_toolbar = TOOLBAR_X1 <= x <= TOOLBAR_X2

            if is_pinch and (now - state.last_click_time) > CLICK_COOLDOWN:
                state.last_click_time = now

                if in_toolbar:
                    for btn in TOOLBAR_BUTTONS:
                        if point_in_button(x, y, btn):
                            activate_button(btn)
                            break
                    state.prev_point = None
                    state.shape_start = None

                else:
                    # Pinch outside toolbar: shape start/end
                    if state.current_shape == "Text":
                        text = input("Enter the text you want to place on screen: ")
                        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, state.current_color, 2)
                        set_status("Text added")
                    elif state.current_shape in ("Line", "Rectangle", "Circle"):
                        if state.shape_start is None:
                            state.shape_start = (x, y)
                            set_status("Start point set; pinch again to set end point")
                        else:
                            draw_final_shape(canvas, state.shape_start, (x, y))
                            state.shape_start = None
                            set_status("Shape drawn")

            elif only_index_up and not in_toolbar:
                # Free drawing or shape preview
                if state.current_shape == "Free":
                    if state.prev_point is not None:
                        cv2.line(canvas, state.prev_point, (x, y),
                                  state.current_color, state.brush_thickness)
                    state.prev_point = (x, y)
                elif state.current_shape == "Eraser":
                    cv2.circle(canvas, (x, y), ERASER_THICKNESS, (0, 0, 0), -1)
                    state.prev_point = (x, y)
                else:
                    state.prev_point = None
            else:
                state.prev_point = None
        else:
            state.prev_point = None

        # Combine canvas with camera frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(bg, fg)

        # Shape preview
        if state.shape_start is not None and index_tip is not None:
            preview = combined.copy()
            draw_final_shape(preview, state.shape_start, index_tip, thickness=2)
            combined = preview

        combined = draw_toolbar(combined)

        # Finger marker
        if index_tip is not None:
            cv2.circle(combined, index_tip, 10,
                       (0, 255, 0) if is_pinch else (255, 255, 255), 2)

        # Status bar
        if state.status_msg and (time.time() - state.status_msg_time) < 2.0:
            cv2.putText(combined, state.status_msg, (20, CAM_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if state.recording:
            cv2.circle(combined, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(combined, "REC", (48, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            if state.video_writer is not None:
                state.video_writer.write(combined)

        cv2.imshow("AI Hand Painter", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_image()
        elif key == ord('r'):
            toggle_recording()
        elif key == ord('c'):
            canvas[:] = 0
            set_status("Canvas cleared")
        elif key == ord('+'):
            state.brush_thickness = min(60, state.brush_thickness + 2)
        elif key == ord('-'):
            state.brush_thickness = max(2, state.brush_thickness - 2)

    if state.video_writer is not None:
        state.video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------
# 10) Shape drawing function
# ----------------------------------------------------------------------
def draw_final_shape(target, p1, p2, thickness=None):
    th = thickness if thickness is not None else state.brush_thickness

    if state.current_shape == "Line":
        cv2.line(target, p1, p2, state.current_color, th)

    elif state.current_shape == "Rectangle":
        cv2.rectangle(target, p1, p2, state.current_color, th)

    elif state.current_shape == "Circle":
        radius = int(distance(p1, p2))
        cv2.circle(target, p1, radius, state.current_color, th)

# ----------------------------------------------------------------------
# 11) Run program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()