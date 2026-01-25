#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirCursor - AI-Powered Virtual Mouse Control
============================================

A comprehensive gesture-based mouse control system using computer vision and AI.
Control your computer mouse cursor using natural hand gestures through your webcam!

Features:
- Cursor control with index finger pointing
- Left/Right click with finger pinch gestures  
- Scroll with peace sign gesture
- Drag & drop with three-finger pinch
- Double click with open palm
- Zoom with two-hand gestures
- Real-time statistics and performance monitoring
- Advanced hand validation (prevents face detection)
- Configurable settings and calibration

Controls:
Point with index finger - Cursor control
Thumb + Index pinch - Left click  
Thumb + Middle pinch - Right click
Peace sign + movement - Scroll up/down
Open palm (5 fingers) - Double click
Three finger pinch - Drag & drop
Two hands spread/pinch - Zoom in/out

Keyboard Controls:
Q/ESC - Quit application
S - Save settings
R - Reset statistics  
C - Calibrate hand size

Installation:
pip install opencv-python mediapipe pyautogui numpy

Author: AirCursor Project
License: MIT
Version: 2.0 Enhanced
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import ctypes
import json
import os
from datetime import datetime

# ===================== SMOOTHING CONFIGURATION =====================
SMOOTHENING = 6   # higher = smoother but slower
prev_x, prev_y = 0, 0

# ===================== CONFIGURATION SYSTEM =====================
class Config:
    """Configuration Class - All Settings in One Place"""
    
    # Webcam Settings
    CAM_WIDTH = 640
    CAM_HEIGHT = 480  
    FRAME_MARGIN = 100
    
    # Performance & Smoothing
    SMOOTHING = 3
    HISTORY_SIZE = 5
    
    # Screen Settings  
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    # Gesture Thresholds (in pixels)
    CLICK_THRESHOLD = 40
    SCROLL_THRESHOLD = 50
    ZOOM_THRESHOLD = 100
    
    # Timing Settings (in seconds)
    CLICK_COOLDOWN = 0.3
    SCROLL_COOLDOWN = 0.1
    GESTURE_COOLDOWN = 0.8
    
    # Detection Settings (0.0-1.0)
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Feature Toggles
    ENABLE_SCROLL = True
    ENABLE_ZOOM = True
    ENABLE_DRAG = True
    ENABLE_DOUBLE_CLICK = True
    ENABLE_GESTURE_TRAIL = True
    ENABLE_STATISTICS = True
    ENABLE_DEBUG_INFO = True

class AppState:
    """Application State Manager"""
    
    def __init__(self):
        # Cursor State
        self.prev_x = 0
        self.prev_y = 0
        self.curr_x = 0
        self.curr_y = 0
        
        # Timing State
        self.last_click_time = 0
        self.last_scroll_time = 0
        self.last_gesture_time = 0
        self.last_double_click_time = 0
        
        # Gesture State
        self.cursor_history = []
        self.is_dragging = False
        self.drag_start_pos = None
        self.scroll_direction = None
        
        # Session Statistics
        self.session_start = time.time()
        self.total_clicks = 0
        self.total_scrolls = 0
        self.total_gestures = 0
        self.frames_processed = 0
        
        # Calibration & Preferences
        self.hand_size_calibration = 1.0
        self.sensitivity = 1.0
        
        # Performance Tracking
        self.last_tick = cv2.getTickCount()
        self.fps_history = []
        
    def update_fps(self):
        """Update FPS calculation"""
        current_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_tick - self.last_tick)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        self.last_tick = current_tick
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_session_duration(self):
        """Get formatted session duration"""
        duration = int(time.time() - self.session_start)
        return f"{duration//60:02d}:{duration%60:02d}"
    
    def reset_statistics(self):
        """Reset session statistics"""
        self.total_clicks = 0
        self.total_scrolls = 0  
        self.total_gestures = 0
        self.frames_processed = 0
        self.session_start = time.time()
        self.fps_history = []

# ===================== MEDIAPIPE INTEGRATION =====================

class MediaPipeCompat:
    """MediaPipe Integration & Compatibility Layer"""
    
    def __init__(self):
        try:
            # Try Legacy MediaPipe API First
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.use_legacy = True
            print("Using Legacy MediaPipe API")
            
        except AttributeError:
            # Use New MediaPipe API
            print("Switching to new MediaPipe API...")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if not exists
            model_path = 'hand_landmarker.task'
            if not os.path.exists(model_path):
                print("Downloading MediaPipe hand landmarker model...")
                try:
                    import urllib.request
                    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Model downloaded successfully")
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    raise
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_legacy = False
            print("Using New MediaPipe API")
    
    def process(self, img_rgb):
        """Process RGB image and return hand landmarks"""
        if self.use_legacy:
            return self.hands.process(img_rgb)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = self.detector.detect(mp_image)
            
            class CompatResult:
                def __init__(self, detection_result):
                    self.multi_hand_landmarks = detection_result.hand_landmarks if detection_result.hand_landmarks else None
            
            return CompatResult(result)
    
    def draw_landmarks(self, img, hand_landmarks):
        """Draw hand landmarks"""
        if self.use_legacy:
            self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            if hasattr(hand_landmarks, 'landmark'):
                landmarks = hand_landmarks.landmark
            else:
                landmarks = hand_landmarks
            
            # Draw landmarks with color coding
            for i, landmark in enumerate(landmarks):
                x = int(landmark.x * Config.CAM_WIDTH)
                y = int(landmark.y * Config.CAM_HEIGHT)
                
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                elif i == 0:  # Wrist
                    cv2.circle(img, (x, y), 8, (255, 0, 0), -1)
                else:  # Other joints
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# ===================== HAND VALIDATION =====================
def is_in_detection_area(hand_landmarks):
    """Check if hand is in valid detection area"""
    if hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    
    wrist = landmarks[0]
    margin = 0.1
    
    return (margin < wrist.x < (1 - margin) and 
            margin < wrist.y < (1 - margin))

def is_valid_hand(hand_landmarks):
    """Advanced hand validation to filter out face features"""
    if hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    
    if len(landmarks) != 21:
        return False
    
    # Calculate hand span
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    hand_span = math.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
    
    # Adaptive threshold based on distance
    distance_factor = 1.0 - wrist.y
    min_span_threshold = 0.08 * max(0.5, distance_factor)
    
    if hand_span < min_span_threshold:
        return False
    
    # Check fingertip structure
    fingertips = [4, 8, 12, 16, 20]
    palm_landmarks = [0, 5, 9, 13, 17]
    palm_center_x = sum(landmarks[i].x for i in palm_landmarks) / len(palm_landmarks)
    palm_center_y = sum(landmarks[i].y for i in palm_landmarks) / len(palm_landmarks)
    
    valid_fingertips = 0
    for tip_idx in fingertips:
        distance = math.hypot(
            landmarks[tip_idx].x - palm_center_x,
            landmarks[tip_idx].y - palm_center_y
        )
        if distance > 0.04:
            valid_fingertips += 1
    
    return valid_fingertips >= 3

def fingers_up(hand_landmarks):
    """Enhanced finger state detection"""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks

    # Thumb detection
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    thumb_extended_x = thumb_tip.x > thumb_ip.x and thumb_tip.x > thumb_mcp.x
    thumb_extended_y = thumb_tip.y < thumb_ip.y and abs(thumb_tip.x - thumb_mcp.x) > 0.02
    thumb_up = thumb_extended_x or thumb_extended_y
    fingers.append(1 if thumb_up else 0)

    # Other fingers
    for i in range(1, 5):
        tip_y = landmarks[tips[i]].y
        pip_y = landmarks[tips[i] - 2].y
        mcp_y = landmarks[tips[i] - 3].y if tips[i] - 3 >= 0 else pip_y
        
        finger_extended = (tip_y < pip_y - 0.01) and (tip_y < mcp_y)
        fingers.append(1 if finger_extended else 0)
    
    return fingers

def get_position(landmarks, idx):
    """Get landmark position with bounds checking"""
    if hasattr(landmarks, 'landmark'):
        landmark_list = landmarks.landmark
    else:
        landmark_list = landmarks
    
    if idx >= len(landmark_list) or idx < 0:
        return 0, 0
    
    landmark = landmark_list[idx]
    x = int(landmark.x * Config.CAM_WIDTH)
    y = int(landmark.y * Config.CAM_HEIGHT)
    
    x = max(0, min(x, Config.CAM_WIDTH - 1))
    y = max(0, min(y, Config.CAM_HEIGHT - 1))
    
    return x, y

# ===================== CURSOR MOVEMENT =====================

def move_cursor(index_x, index_y, state):
    """Simple cursor movement with toggleable smoothing"""
    global prev_x, prev_y
    
    # Map to screen coordinates
    x = int(np.interp(
        index_x, 
        (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN), 
        (0, Config.SCREEN_WIDTH)
    ))
    y = int(np.interp(
        index_y, 
        (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN), 
        (0, Config.SCREEN_HEIGHT)
    ))
    
    # Apply smoothing
    smooth_x = prev_x + (x - prev_x) // SMOOTHENING
    smooth_y = prev_y + (y - prev_y) // SMOOTHENING
    
    prev_x, prev_y = smooth_x, smooth_y
    
    # Move cursor safely
    try:
        smooth_x = max(0, min(smooth_x, Config.SCREEN_WIDTH - 1))
        smooth_y = max(0, min(smooth_y, Config.SCREEN_HEIGHT - 1))
        
        pyautogui.moveTo(smooth_x, smooth_y)
        
        # Update state for compatibility
        state.curr_x, state.curr_y = smooth_x, smooth_y
        state.prev_x, state.prev_y = smooth_x, smooth_y
        
    except Exception as e:
        if Config.ENABLE_DEBUG_INFO:
            print(f"Cursor movement error: {e}")

# ===================== GESTURE HANDLERS =====================
def handle_scroll_gesture(hand_landmarks, state, img):
    """Handle scroll gestures with peace sign"""
    if not Config.ENABLE_SCROLL:
        return False
    
    fingers = fingers_up(hand_landmarks)
    now = time.time()
    
    # Peace sign (index + middle up, others down)
    if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
        index_x, index_y = get_position(hand_landmarks, 8)
        middle_x, middle_y = get_position(hand_landmarks, 12)
        
        avg_y = (index_y + middle_y) / 2
        
        if state.scroll_direction is None:
            state.scroll_direction = avg_y
            return True
        
        if now - state.last_scroll_time > Config.SCROLL_COOLDOWN:
            y_movement = state.scroll_direction - avg_y
            
            if abs(y_movement) > 20:
                if y_movement > 0:
                    pyautogui.scroll(3)
                    cv2.putText(img, "SCROLL UP", (200, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    pyautogui.scroll(-3)
                    cv2.putText(img, "SCROLL DOWN", (200, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                state.scroll_direction = avg_y
                state.last_scroll_time = now
                state.total_scrolls += 1
        
        cv2.circle(img, (index_x, index_y), 12, (0, 255, 255), 3)
        cv2.circle(img, (middle_x, middle_y), 12, (0, 255, 255), 3)
        cv2.line(img, (index_x, index_y), (middle_x, middle_y), (0, 255, 255), 3)
        
        return True
    else:
        state.scroll_direction = None
        return False

def handle_zoom_gesture(hand_landmarks_list, state, img):
    """Handle zoom gestures with two hands"""
    if not Config.ENABLE_ZOOM or len(hand_landmarks_list) < 2:
        return False
    
    hand1_index = get_position(hand_landmarks_list[0], 8)
    hand2_index = get_position(hand_landmarks_list[1], 8)
    
    current_distance = math.hypot(
        hand1_index[0] - hand2_index[0], 
        hand1_index[1] - hand2_index[1]
    )
    
    if hasattr(state, 'prev_zoom_distance'):
        distance_change = current_distance - state.prev_zoom_distance
        
        if abs(distance_change) > 30:
            if distance_change > 0:
                pyautogui.hotkey('ctrl', '+')
                cv2.putText(img, "ZOOM IN", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                pyautogui.hotkey('ctrl', '-')
                cv2.putText(img, "ZOOM OUT", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            state.prev_zoom_distance = current_distance
    else:
        state.prev_zoom_distance = current_distance
    
    cv2.circle(img, hand1_index, 15, (255, 0, 255), 3)
    cv2.circle(img, hand2_index, 15, (255, 0, 255), 3)
    cv2.line(img, hand1_index, hand2_index, (255, 0, 255), 2)
    
    return True

def handle_drag_gesture(hand_landmarks, state, img):
    """Handle drag and drop gestures"""
    if not Config.ENABLE_DRAG:
        return False
    
    thumb_x, thumb_y = get_position(hand_landmarks, 4)
    index_x, index_y = get_position(hand_landmarks, 8)
    middle_x, middle_y = get_position(hand_landmarks, 12)
    
    thumb_index_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
    thumb_middle_dist = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
    index_middle_dist = math.hypot(middle_x - index_x, middle_y - index_y)
    
    three_finger_pinch = (thumb_index_dist < 35 and 
                         thumb_middle_dist < 35 and 
                         index_middle_dist < 25)
    
    if three_finger_pinch:
        if not state.is_dragging:
            pyautogui.mouseDown()
            state.is_dragging = True
            state.drag_start_pos = (index_x, index_y)
            cv2.putText(img, "DRAG START", (200, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            move_cursor(index_x, index_y, state)
            cv2.putText(img, "DRAGGING", (200, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.circle(img, (thumb_x, thumb_y), 12, (255, 255, 0), -1)
        cv2.circle(img, (index_x, index_y), 12, (255, 255, 0), -1)
        cv2.circle(img, (middle_x, middle_y), 12, (255, 255, 0), -1)
        
        return True
    else:
        if state.is_dragging:
            pyautogui.mouseUp()
            state.is_dragging = False
            state.drag_start_pos = None
            cv2.putText(img, "DRAG END", (200, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return False

def handle_double_click(hand_landmarks, state, img):
    """Handle double click with open palm"""
    if not Config.ENABLE_DOUBLE_CLICK:
        return False
    
    fingers = fingers_up(hand_landmarks)
    now = time.time()
    
    # Open palm (all 5 fingers up)
    if sum(fingers) == 5:
        if now - state.last_double_click_time > 1.0:
            pyautogui.doubleClick()
            state.last_double_click_time = now
            state.total_clicks += 2
            state.total_gestures += 1
            
            cv2.putText(img, "DOUBLE CLICK", (200, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return True
    
    return False

# ===================== USER INTERFACE =====================

def draw_enhanced_ui(img, state, valid_hands_count):
    """Draw enhanced user interface"""
    # Detection area boundary
    margin_x = int(0.1 * Config.CAM_WIDTH)
    margin_y = int(0.1 * Config.CAM_HEIGHT)
    
    cv2.rectangle(img, (margin_x, margin_y), 
                 (Config.CAM_WIDTH - margin_x, Config.CAM_HEIGHT - margin_y), 
                 (100, 100, 100), 2)
    
    cv2.putText(img, "Hand Detection Area", (margin_x + 10, margin_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    if not Config.ENABLE_STATISTICS:
        return
    
    # Statistics panel
    panel_width, panel_height = 320, 160
    cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
    
    # Session statistics
    session_duration = state.get_session_duration()
    avg_fps = state.update_fps()
    
    stats_lines = [
        f"Session: {session_duration}",
        f"Hands: {valid_hands_count}/2",
        f"Clicks: {state.total_clicks}",
        f"Scrolls: {state.total_scrolls}",
        f"Gestures: {state.total_gestures}",
        f"FPS: {avg_fps:.1f}",
        f"Frames: {state.frames_processed}"
    ]
    
    for i, text in enumerate(stats_lines):
        y_pos = 30 + i * 18
        cv2.putText(img, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Feature status indicators
    features_y_start = Config.CAM_HEIGHT - 120
    
    feature_status = [
        ("Scroll", Config.ENABLE_SCROLL),
        ("Zoom", Config.ENABLE_ZOOM),
        ("Drag", Config.ENABLE_DRAG),
        ("2xClick", Config.ENABLE_DOUBLE_CLICK),
        ("Trail", Config.ENABLE_GESTURE_TRAIL),
        ("Debug", Config.ENABLE_DEBUG_INFO)
    ]
    
    for i, (feature_name, enabled) in enumerate(feature_status):
        x_pos = 10 + (i % 3) * 100
        y_pos = features_y_start + (i // 3) * 25
        
        color = (0, 255, 0) if enabled else (0, 0, 255)
        status_text = "ON" if enabled else "OFF"
        
        cv2.putText(img, f"{feature_name}: {status_text}", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Performance indicator
    perf_color = (0, 255, 0) if avg_fps > 20 else (0, 255, 255) if avg_fps > 15 else (0, 0, 255)
    perf_status = "EXCELLENT" if avg_fps > 25 else "GOOD" if avg_fps > 20 else "FAIR" if avg_fps > 15 else "POOR"
    
    cv2.putText(img, f"Performance: {perf_status}", (Config.CAM_WIDTH - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)

def draw_gesture_trail(img, state):
    """Draw gesture trail visualization"""
    if not Config.ENABLE_GESTURE_TRAIL or len(state.cursor_history) < 2:
        return
    
    for i in range(1, len(state.cursor_history)):
        if i >= len(state.cursor_history):
            break
        
        prev_pos = state.cursor_history[i-1]
        curr_pos = state.cursor_history[i]
        
        # Convert screen coordinates back to camera coordinates
        prev_cam_x = int(np.interp(prev_pos[0], (0, Config.SCREEN_WIDTH), 
                                  (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN)))
        prev_cam_y = int(np.interp(prev_pos[1], (0, Config.SCREEN_HEIGHT), 
                                  (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN)))
        
        curr_cam_x = int(np.interp(curr_pos[0], (0, Config.SCREEN_WIDTH), 
                                  (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN)))
        curr_cam_y = int(np.interp(curr_pos[1], (0, Config.SCREEN_HEIGHT), 
                                  (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN)))
        
        # Fading effect
        alpha = i / len(state.cursor_history)
        color = (int(255 * alpha), int(100 * alpha), int(200 * alpha))
        thickness = max(1, int(3 * alpha))
        cv2.line(img, (prev_cam_x, prev_cam_y), (curr_cam_x, curr_cam_y), color, thickness)

# ===================== SETTINGS SYSTEM =====================
def save_settings(state):
    """Save settings and statistics"""
    settings_data = {
        'statistics': {
            'total_clicks': state.total_clicks,
            'total_scrolls': state.total_scrolls,
            'total_gestures': state.total_gestures,
            'frames_processed': state.frames_processed,
            'session_duration': state.get_session_duration()
        },
        'calibration': {
            'hand_size_calibration': state.hand_size_calibration,
            'sensitivity': state.sensitivity,
        },
        'config': {
            'smoothing': Config.SMOOTHING,
            'click_threshold': Config.CLICK_THRESHOLD,
            'scroll_threshold': Config.SCROLL_THRESHOLD,
        },
        'metadata': {
            'last_session': datetime.now().isoformat(),
            'version': '2.0 Enhanced',
            'save_count': getattr(state, 'save_count', 0) + 1
        }
    }
    
    try:
        settings_file = 'aircursor_settings.json'
        if os.path.exists(settings_file):
            backup_file = f'aircursor_settings_backup_{int(time.time())}.json'
            os.rename(settings_file, backup_file)
        
        with open(settings_file, 'w') as f:
            json.dump(settings_data, f, indent=2)
        
        state.save_count = settings_data['metadata']['save_count']
        print(f"Settings saved successfully (Save #{state.save_count})")
        
    except Exception as e:
        print(f"Failed to save settings: {e}")

def load_settings(state):
    """Load saved settings"""
    try:
        with open('aircursor_settings.json', 'r') as f:
            settings_data = json.load(f)
        
        if 'statistics' in settings_data:
            stats = settings_data['statistics']
            state.total_clicks = stats.get('total_clicks', 0)
            state.total_scrolls = stats.get('total_scrolls', 0)
            state.total_gestures = stats.get('total_gestures', 0)
        
        if 'calibration' in settings_data:
            cal = settings_data['calibration']
            state.hand_size_calibration = cal.get('hand_size_calibration', 1.0)
            state.sensitivity = cal.get('sensitivity', 1.0)
        
        print("Settings loaded successfully")
        return True
        
    except FileNotFoundError:
        print("No previous settings found, using defaults")
        return False
    except Exception as e:
        print(f"Error loading settings: {e}")
        return False

def calibrate_hand_size(state):
    """Hand size calibration"""
    print("Hand Size Calibration")
    print("Instructions:")
    print("   1. Show your hand in normal position")
    print("   2. Keep hand steady for 3 seconds")
    print("   3. Calibration will complete automatically")
    
    state.hand_size_calibration = 1.0
    print("Calibration complete (using default values)")

# ===================== MAIN APPLICATION =====================

def main():
    """Main application entry point"""
    print("AirCursor Enhanced - AI-Powered Virtual Mouse")
    print("=" * 50)
    
    try:
        # Initialize components
        state = AppState()
        load_settings(state)
        
        # Setup camera
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Camera not accessible")
            print("Troubleshooting:")
            print("   - Check camera permissions")
            print("   - Close other applications using camera")
            print("   - Try different camera index (1, 2, etc.)")
            return
        
        print("Camera initialized successfully")
        
        # Optimize PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        # Initialize MediaPipe
        print("Initializing MediaPipe hand detection...")
        mp_compat = MediaPipeCompat()
        
        # Setup window
        window_name = "AirCursor Enhanced - AI Virtual Mouse"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        try:
            cv2.moveWindow(window_name, 100, 100)
        except:
            pass
        
        print("Initialization complete!")
        print()
        print("GESTURE CONTROLS:")
        print("   Point with index finger -> Move cursor")
        print("   Thumb + Index pinch -> Left click")
        print("   Thumb + Middle pinch -> Right click")
        print("   Peace sign + movement -> Scroll up/down")
        print("   Open palm (5 fingers) -> Double click")
        print("   Three finger pinch -> Drag & drop")
        print("   Two hands spread/pinch -> Zoom in/out")
        print()
        print("KEYBOARD CONTROLS:")
        print("   Q or ESC -> Quit application")
        print("   S -> Save current settings")
        print("   R -> Reset session statistics")
        print("   C -> Calibrate hand size")
        print()
        print("Starting AirCursor... Show your hand to begin!")
        print("=" * 50)
        
        # Runtime variables
        show_help = False
        paused = False
        
    except Exception as e:
        print(f"Initialization error: {e}")
        return
    
    # Main loop
    try:
        while True:
            # Capture frame
            success, img = cap.read()
            if not success:
                print("Failed to read camera frame")
                continue
            
            img = cv2.flip(img, 1)
            state.frames_processed += 1
            
            # Process for hand detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_compat.process(img_rgb)
            
            # Draw UI
            draw_enhanced_ui(img, state, 0)
            
            # Hand processing
            if results.multi_hand_landmarks and not paused:
                valid_hands = []
                
                # Validate hands
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_valid_hand(hand_landmarks) and is_in_detection_area(hand_landmarks):
                        valid_hands.append(hand_landmarks)
                
                draw_enhanced_ui(img, state, len(valid_hands))
                
                if valid_hands:
                    # Process primary hand
                    primary_hand = valid_hands[0]
                    mp_compat.draw_landmarks(img, primary_hand)
                    
                    # Extract hand data
                    fingers = fingers_up(primary_hand)
                    index_x, index_y = get_position(primary_hand, 8)
                    thumb_x, thumb_y = get_position(primary_hand, 4)
                    middle_x, middle_y = get_position(primary_hand, 12)
                    
                    now = time.time()
                    
                    # Debug info
                    if Config.ENABLE_DEBUG_INFO:
                        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                        for i, (name, status) in enumerate(zip(finger_names, fingers)):
                            color = (0, 255, 0) if status else (0, 0, 255)
                            cv2.putText(img, f"{name}: {'UP' if status else 'DN'}", 
                                       (350, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Calculate distances
                    thumb_index_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    thumb_middle_dist = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
                    
                    if Config.ENABLE_DEBUG_INFO:
                        cv2.putText(img, f"T-I: {thumb_index_dist:.1f}px", (350, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img, f"T-M: {thumb_middle_dist:.1f}px", (350, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Gesture recognition (priority order)
                    gesture_detected = False
                    
                    # Priority 1: Drag gesture
                    if handle_drag_gesture(primary_hand, state, img):
                        gesture_detected = True
                    
                    # Priority 2: Double click
                    elif handle_double_click(primary_hand, state, img):
                        gesture_detected = True
                    
                    # Priority 3: Scroll gesture
                    elif handle_scroll_gesture(primary_hand, state, img):
                        gesture_detected = True
                    
                    # Priority 4: Click gestures
                    elif thumb_index_dist < Config.CLICK_THRESHOLD:
                        # Left click
                        if now - state.last_click_time > Config.CLICK_COOLDOWN:
                            pyautogui.click()
                            state.last_click_time = now
                            state.total_clicks += 1
                        
                        cv2.putText(img, "LEFT CLICK", (450, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.circle(img, (thumb_x, thumb_y), 20, (0, 255, 0), 3)
                        cv2.circle(img, (index_x, index_y), 20, (0, 255, 0), 3)
                        cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
                        gesture_detected = True
                    
                    elif thumb_middle_dist < Config.CLICK_THRESHOLD:
                        # Right click
                        if now - state.last_click_time > Config.CLICK_COOLDOWN:
                            pyautogui.click(button='right')
                            state.last_click_time = now
                            state.total_clicks += 1
                        
                        cv2.putText(img, "RIGHT CLICK", (450, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.circle(img, (thumb_x, thumb_y), 20, (0, 0, 255), 3)
                        cv2.circle(img, (middle_x, middle_y), 20, (0, 0, 255), 3)
                        cv2.line(img, (thumb_x, thumb_y), (middle_x, middle_y), (0, 0, 255), 3)
                        gesture_detected = True
                    
                    # Priority 5: Cursor movement
                    elif fingers[1] == 1 and not gesture_detected:
                        move_cursor(index_x, index_y, state)
                        cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), -1)
                        cv2.putText(img, "CURSOR", (450, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Two-hand gestures
                    if len(valid_hands) >= 2:
                        handle_zoom_gesture(valid_hands, state, img)
                    
                    # Draw gesture trail
                    draw_gesture_trail(img, state)
            
            # Display image
            cv2.imshow(window_name, img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                print("Quitting AirCursor...")
                break
                
            elif key == ord('s'):  # Save settings
                save_settings(state)
                
            elif key == ord('r'):  # Reset statistics
                state.reset_statistics()
                print("Session statistics reset")
                
            elif key == ord('c'):  # Calibrate
                calibrate_hand_size(state)
                
            elif key == ord('p'):  # Pause/unpause
                paused = not paused
                print(f"AirCursor: {'PAUSED' if paused else 'RESUMED'}")
                
            elif key == ord('d'):  # Toggle debug
                Config.ENABLE_DEBUG_INFO = not Config.ENABLE_DEBUG_INFO
                print(f"Debug info: {'ON' if Config.ENABLE_DEBUG_INFO else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nAirCursor interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"Runtime error: {e}")
        if Config.ENABLE_DEBUG_INFO:
            import traceback
            traceback.print_exc()
    
    finally:
        # Cleanup
        print("Cleaning up resources...")
        
        try:
            save_settings(state)
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"Final Session Statistics:")
            print(f"   Duration: {state.get_session_duration()}")
            print(f"   Clicks: {state.total_clicks}")
            print(f"   Scrolls: {state.total_scrolls}")
            print(f"   Gestures: {state.total_gestures}")
            print(f"   Frames: {state.frames_processed}")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("AirCursor Enhanced closed successfully")
        print("Thank you for using AirCursor!")

def print_installation_guide():
    """Installation guide"""
    guide = """
AirCursor Enhanced - Installation Guide
======================================

SYSTEM REQUIREMENTS:
- Python 3.7 or higher
- Webcam (built-in or USB)
- 4GB RAM (8GB recommended)
- Windows 10/11, macOS 10.14+, or Ubuntu 18.04+

INSTALLATION STEPS:

1. Install Python Dependencies:
   pip install opencv-python mediapipe pyautogui numpy

2. Download AirCursor:
   - Save this file as 'AirCursor_Clean.py'
   - Ensure camera permissions are granted

3. Run AirCursor:
   python AirCursor_Clean.py

TROUBLESHOOTING:

Camera Not Opening:
   - Check camera permissions in system settings
   - Close other applications using camera
   - Try different camera index: cv2.VideoCapture(1)

MediaPipe Import Error:
   pip install --upgrade mediapipe

PyAutoGUI Permission Error (macOS):
   - System Preferences -> Security & Privacy -> Accessibility
   - Add Terminal or Python to allowed applications

Poor Performance:
   - Close unnecessary applications
   - Reduce camera resolution in Config class
   - Disable gesture trail: Config.ENABLE_GESTURE_TRAIL = False

OPTIMIZATION TIPS:
- Use good lighting for better hand detection
- Keep hand in center detection area (gray rectangle)
- Avoid cluttered backgrounds
- Calibrate hand size for optimal detection

ENJOY USING AIRCURSOR!
"""
    print(guide)

# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    """Application entry point"""
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            print_installation_guide()
            sys.exit(0)
            
        elif arg in ['--version', '-v', 'version']:
            print("AirCursor Enhanced v2.0")
            print("AI-Powered Virtual Mouse Control")
            print("Built with OpenCV, MediaPipe, and PyAutoGUI")
            sys.exit(0)
            
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for installation guide")
            sys.exit(1)
    
    # Start the application
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages:")
        print("   pip install opencv-python mediapipe pyautogui numpy")
        print("Use --help for complete installation guide")
    except Exception as e:
        print(f"Failed to start AirCursor: {e}")
        print("Use --help for troubleshooting guide")