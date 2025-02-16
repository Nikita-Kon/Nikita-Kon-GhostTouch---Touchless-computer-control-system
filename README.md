# 🖐️ Hand Tracking Mouse Control

This project utilizes **OpenCV, Flask, and MediaPipe** to track hand movements and control the mouse cursor based on detected hand gestures. The program processes video frames in real-time, identifies key hand landmarks, and translates them into mouse movement and clicks.

---

## 📦 Dependencies

Install the required dependencies using:
```sh
pip install flask flask-cors numpy opencv-python mediapipe mouse
```

### Required Libraries:
- `Flask`
- `Flask-CORS`
- `NumPy`
- `OpenCV`
- `MediaPipe`
- `Mouse`
- `Threading`
- `Math`
- `Time`

---

## 📂 Project Structure

### 🖥️ Flask Server
- Receives video frames and updates the latest frame.

### ✋ Hand Tracking
- Uses **MediaPipe** to detect hand landmarks and analyze gestures.

### 🖱️ Mouse Control
- Maps hand movement to screen coordinates and simulates mouse clicks.

---

## 🚀 Server Setup

The Flask server listens for incoming video frames:

### API Endpoints:
- `/` - Serves the index page.
- `/send` - Receives a video frame via POST request.

### Flask Server Code:
```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def receive_frame():
    global latest_frame
    if 'frame' not in request.files:
        return "No frame received", 400
    file = request.files['frame']
    np_arr = np.frombuffer(file.read(), np.uint8)
    latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return "Frame received", 200
```

---

## 🖐️ Hand Tracking and Mouse Control

### 🔑 Key Functions:
```python
def smooth(prev, curr):
    """Applies exponential smoothing to reduce noise."""
    return prev * 0.8 + curr * 0.2

def moving_average(prev_values, curr_value, window_size=5):
    """Computes the moving average of the last window_size values."""
    prev_values.append(curr_value)
    if len(prev_values) > window_size:
        prev_values.pop(0)
    return sum(prev_values) / len(prev_values)

def map_range(value, minIn, maxIn, minOut, maxOut):
    """Maps a value from one range to another."""
    return minOut + (value - minIn) * (maxOut - minOut) / (maxIn - minIn)
```

### 🎮 Gesture Control:
- **Moving the hand** → Moves the mouse cursor.
- **Pinching index finger and thumb** → Simulates a left-click.
- **Holding a gesture** → Simulates a mouse hold.

```python
if length_for_click < 0.05:
    mouse.click(button='left')
    time.sleep(0.3)
```

---

## ⚙️ Execution Modes

- **Native Mode**: Uses the local webcam.
- **Web Mode**: Receives frames from another device via Flask.

```python
Mode = "Native"
if Mode == "Native":
    while cap.isOpened():
        success, image = cap.read()
        MoveByHand(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
elif Mode == "Web":
    threading.Thread(target=MoveByHandWeb, daemon=False).start()
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
```

---

## ▶️ Running the Program

Run the script:
```sh
python script.py
```

### Select Mode:
- `"Native"` → Uses the local webcam.
- `"Web"` → Receives frames remotely.

### Exit:
- Press `q` to terminate.

---

## 📝 Notes
- Ensure that the **webcam is properly connected**.
- Adjust confidence thresholds if **tracking is unreliable**.
- Use `cv2.imshow()` to visualize the processed frames.

---

## 🔮 Future Improvements
- ✅ Multi-hand tracking.
- ✅ Enhanced gesture recognition.
- ✅ Improved smoothing for more stable cursor movement.

---


