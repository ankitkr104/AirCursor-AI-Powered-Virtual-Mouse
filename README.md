Virtual Mouse Using Hand Gestures (Python)

Control your computer mouse using hand gestures and a webcam — no physical mouse required.
This project blends computer vision and AI-based hand tracking to convert finger movements into real-time mouse actions.

Your hand becomes the controller.
The webcam becomes the sensor.
Science fiction, politely applied.

📌 Overview

The Virtual Mouse detects hand gestures through a webcam and maps them to mouse operations such as cursor movement, left click, and right click.

It uses MediaPipe Hand Landmarks for accurate finger tracking and PyAutoGUI to control the system mouse smoothly and reliably.

✨ Features

🖐️ Cursor movement using index finger

👆 Left click using thumb + index finger

👉 Right click using thumb + middle finger

✌️ YES gesture detection (index + middle finger up)

🎥 Real-time hand tracking via webcam

🪟 Always-on-top camera preview window

🎯 Smooth cursor motion with jitter reduction

⏱️ Click cooldown to prevent accidental clicks

🛑 Press ESC to safely exit the program

🛠️ Technologies Used

Python – Core programming language

OpenCV – Webcam access and image processing

MediaPipe – AI-based hand landmark detection

PyAutoGUI – Mouse control automation

NumPy – Numerical calculations

Math & Time – Gesture distance and cooldown handling

ctypes – Windows window behavior control

✋ Hand Gestures & Actions
Gesture	Action
Index finger up	Move mouse cursor
Thumb + Index close	Left click
Thumb + Middle close	Right click
Index + Middle up	YES gesture detected
ESC key	Exit program
🔄 How It Works (Simple Flow)

Webcam captures real-time video

MediaPipe detects 21 hand landmarks

Finger states (up/down) are analyzed

Landmark positions are mapped to screen coordinates

PyAutoGUI performs mouse actions

Smoothing and cooldown logic ensure stability

🚀 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/yourusername/virtual-mouse.git
cd virtual-mouse

2️⃣ Install Dependencies
pip install opencv-python mediapipe pyautogui numpy


⚠️ Make sure your webcam is connected and accessible.

▶️ Run the Project
python virtual_mouse.py


A window titled “Virtual Mouse” will appear

Show your hand clearly in front of the webcam

Press ESC to exit safely

⚙️ Customization

You can tune performance by modifying these values in the code:

SMOOTHING = 5        # Cursor smoothness
FRAME_MARGIN = 50   # Hand movement boundary
CLICK_COOLDOWN = 0.5  # Delay between clicks


Increase SMOOTHING for steadier cursor movement

Adjust FRAME_MARGIN for a comfortable hand range

🧪 Tested On

Windows 10 / 11

Python 3.9+

Built-in laptop webcam

🌱 Future Improvements

Scroll gestures

Drag & drop support

Gesture-based virtual keyboard

Multi-hand detection

Cross-platform support (Linux / macOS)

🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request to improve the project.

📜 License

Open-source and free to use for learning, experimentation, and personal projects.

🧠 Fun Fact

Your hand becomes a biological joystick, and the webcam becomes its nervous system.
Cyberpunk? Perhaps.
Useful? Definitely.
