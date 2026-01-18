**Virtual Mouse using Hand Gestures (Python)**

This project allows you to control your computer mouse using hand gestures through a webcam.
No physical mouse required — just your hand and a camera.

It uses computer vision + AI hand tracking to detect finger movements and convert them into mouse actions like move, left click, and right click.

**Features**

🖱️ Move mouse cursor using index finger.
👆 Left click using thumb + index finger.
👉 Right click using thumb + middle finger.
✅ Detects YES gesture (index + middle finger up).
🎥 Real-time hand tracking via webcam.
🪟 Camera preview window stays always on top.
🧠 Smooth cursor movement (no shaky motion).
🛠️ Technologies Used

**Python**

OpenCV – webcam access & image processing
MediaPipe – AI-powered hand landmark detection
PyAutoGUI – mouse control
NumPy – mathematical operations
Math & Time – gesture distance & cooldown handling
ctypes – Windows system window control

 **Hand Gestures & Actions**
Gesture	Action
Index finger up only	Move cursor
Thumb + Index close	Left click
Thumb + Middle close	Right click
Index + Middle up	YES detected
ESC key	Exit program
📷 How It Works (Simple Explanation)

Webcam captures live video

MediaPipe detects hand landmarks (21 points)

Program checks which fingers are up/down

Finger positions are mapped to screen coordinates

PyAutoGUI moves the mouse or performs clicks

Cooldown prevents accidental multiple clicks

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/virtual-mouse.git
cd virtual-mouse

2️⃣ Install required libraries
pip install opencv-python mediapipe pyautogui numpy


⚠️ Make sure your webcam is connected and working.

▶️ Run the Project
python virtual_mouse.py


A small window titled "Virtual Mouse" will appear

Show your hand in front of the camera

Press ESC to exit

⚙️ Customization

You can adjust these values in the code:

SMOOTHING = 5          # Cursor smoothness
FRAME_MARGIN = 50     # Hand movement boundary
CLICK_COOLDOWN = 0.5  # Click delay


Increase smoothing for slower, steadier movement.

🧪 Tested On

Windows 10 / 11

Python 3.9+

Built-in laptop webcam

🌱 Future Improvements

Scroll using hand gesture

Drag & drop functionality

Gesture-based keyboard control

Multi-hand support

Cross-platform optimization (Linux / macOS)

🤝 Contribution

Contributions, issues, and feature requests are welcome.
Feel free to open a pull request or issue.

📜 License

This project is open-source and free to use for learning and personal projects.

🧠 Fun Fact

Your hand becomes a biological joystick, and the webcam becomes its nervous system.
Cyberpunk? Maybe. Useful? Definitely.

If you want, I can also:

shorten it for GitHub

add images/diagrams

make a professional resume-ready version

convert it to Markdown with badges
