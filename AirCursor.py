import cv2  # Unlocks tools to read, show, edit, and analyze images/videos. Show images, Use webcam, Detect faces, Add filters (blur, edges, etc.)
import mediapipe as mp # Is used when you want AI-powered tracking (hands, face, body) instead of just normal image processing (which cv2 does).
import pyautogui # Is used to control the mouse, keyboard, and screen automatically with Python. It is like teaching Python to “act like a human” on your computer.
import numpy as np # NumPy (Numerical Python) is a library for working with numbers, arrays, and mathematics.
import time # Is used whenever you want to pause, measure, or display time in your Python program.
import math # It is a built-in Python module that gives access to many mathematical functions and constants like square root, trigonometry, logarithms, factorial, π (pi), etc.
import ctypes # It is used when you need speed, low-level access, or system calls that normal Python cannot do directly.

# Webcam resolution
CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_MARGIN = 50
SMOOTHING = 5

# Screen resolution
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Click cooldown (seconds)
CLICK_COOLDOWN = 0.5
YES_COOLDOWN = 1

# ===================== State Variables =====================
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
last_click_time = 0
last_yes_time = 0

# ===================== Setup Webcam =====================
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

if not cap.isOpened():
    print("❌ Camera not opened.")
    exit()

# ===================== Setup MediaPipe =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ===================== Setup Window (Always on Top) =====================
cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Mouse", 320, 240)
cv2.moveWindow("Virtual Mouse", 1200, 0)

hwnd = ctypes.windll.user32.FindWindowW(None, "Virtual Mouse")
ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

# ===================== Helper Functions =====================
def fingers_up(hand_landmarks):
    """Returns list of 5 elements (1=up, 0=down) for Thumb, Index, Middle, Ring, Pinky."""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if hand_landmarks.landmark[tips[0]].x <
                    hand_landmarks.landmark[tips[0] - 1].x else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tips[i]].y <
                        hand_landmarks.landmark[tips[i] - 2].y else 0)
    return fingers

def get_position(landmarks, idx):
    """Returns (x, y) position of landmark in webcam coordinates."""
    return int(landmarks.landmark[idx].x * CAM_WIDTH), int(landmarks.landmark[idx].y * CAM_HEIGHT)

def move_cursor(index_x, index_y):
    """Moves the cursor smoothly based on index finger position."""
    global prev_x, prev_y, curr_x, curr_y
    target_x = np.interp(index_x, (FRAME_MARGIN, CAM_WIDTH - FRAME_MARGIN), (0, SCREEN_WIDTH))
    target_y = np.interp(index_y, (FRAME_MARGIN, CAM_HEIGHT - FRAME_MARGIN), (0, SCREEN_HEIGHT))

    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
    curr_y = prev_y + (target_y - prev_y) / SMOOTHING

    pyautogui.moveTo(curr_x, curr_y)
    prev_x, prev_y = curr_x, curr_y
# ===================== Main Loop =====================
while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read frame")
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)
            index_x, index_y = get_position(hand_landmarks, 8)
            thumb_x, thumb_y = get_position(hand_landmarks, 4)
            middle_x, middle_y = get_position(hand_landmarks, 12)

            now = time.time()

            # Cursor movement: Only Index up
            if fingers[1] == 1 and sum(fingers) == 1:
                move_cursor(index_x, index_y)

            # Left Click: Thumb + Index close
            if math.hypot(index_x - thumb_x, index_y - thumb_y) < 40 and fingers[0] == 1 and fingers[1] == 1:
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    print("🖱️ Left Click")
                    last_click_time = now

            # Right Click: Thumb + Middle close
            if math.hypot(middle_x - thumb_x, middle_y - thumb_y) < 40 and fingers[0] == 1 and fingers[2] == 1:
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click(button='right')
                    print("🖱️ Right Click")
                    last_click_time = now

            # YES Gesture: Index + Middle up only
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                if now - last_yes_time > YES_COOLDOWN:
                    print("✅ YES Detected!")
                    last_yes_time = now

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break