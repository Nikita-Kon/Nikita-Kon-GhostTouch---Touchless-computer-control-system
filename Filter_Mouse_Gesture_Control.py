# Import necessary modules
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
import threading
import time
import cv2
import math
import mediapipe as mp
import mouse

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Last received web frame
latest_frame = None

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to receive frames from a client
@app.route('/send', methods=['POST'])
def receive_frame():
    global latest_frame

    if 'frame' not in request.files:
        return "No frame received", 400

    file = request.files['frame']
    np_arr = np.frombuffer(file.read(), np.uint8)
    latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return "Frame received", 200

# solution hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# global result variables
hold_state = False

def smooth(prev, curr):
    """Applies exponential smoothing to a value."""
    return alpha * curr + (1 - alpha) * prev

# finds average of window size arr
def moving_average(prev_values, curr_value):
    """Calculates the moving average of a list of values."""
    if len(prev_values) == window_size:
        prev_values.pop(0)
    prev_values.append(curr_value)
    return sum(prev_values) / len(prev_values)

def map_range(value, minIn, maxIn, minOut, maxOut):
    """Maps a value from one range to another."""
    return (value - minIn) * (maxOut - minOut) / (maxIn - minIn)

def MoveByHand(image):
    """Processes the frame to detect hand landmarks and control mouse movement."""
    global hold_state

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Detect hand landmarks
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Process detected hand landmarks
            if hand_landmarks.landmark:
                finger8_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger4_point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                cursor_landmarkX = (finger8_point.x + finger4_point.x) / 2
                cursor_landmarkY = (finger4_point.y + finger8_point.y) / 2

                # Convert normalized landmark positions to pixel values
                xhand = int((cursor_landmarkX * 640))
                yhand = int(cursor_landmarkY * 480)

                # Clamp values within a predefined range
                xhand = min(max(xhand, 100), 540)
                yhand = min(max(yhand, 85), 395)

                # Map hand coordinates to screen coordinates
                xhand = map_range(xhand, 100, 540, 0, 1920)
                yhand = map_range(yhand, 85, 395, 0, 1080)

                # Get current mouse position
                cord_mouse = mouse.get_position()
                xmouse = cord_mouse[0]
                ymouse = cord_mouse[1]

                # Draw bounding box for hand tracking region
                cv2.line(image, (100, 85), (540, 85), (0, 255, 0), 1)
                cv2.line(image, (540, 85), (540, 395), (0, 255, 0), 1)
                cv2.line(image, (100, 85), (100, 395), (0, 255, 0), 1)
                cv2.line(image, (100, 395), (540, 395), (0, 255, 0), 1)

                # Apply smoothing using moving average
                prev_x_values = []
                prev_y_values = []
                xhand = moving_average(prev_x_values, xhand)
                yhand = moving_average(prev_y_values, yhand)

                # Exponential filtering of movement
                if len(arr) == 0 or len(arr) == 1:
                    arr.append([xhand, yhand])
                else:
                    smoothed_x = smooth(arr[0][0], arr[1][0])
                    smoothed_y = smooth(arr[0][1], arr[1][1])
                    arr[0] = [smoothed_x, smoothed_y]
                    arr[1] = [xhand, yhand]  # Assign a new list to arr[1]

                # Get Finger Points
                finger8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                finger5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                finger12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate distances for movement, clicking, and holding
                length_for_move = math.sqrt(((finger8.x - finger4.x) ** 2) + ((finger8.y - finger4.y) ** 2))
                length_for_click = math.sqrt(((finger4.x - finger5.x) ** 2) + ((finger4.y - finger5.y) ** 2))
                length_for_hold = math.sqrt(((finger12.x - finger4.x) ** 2) + ((finger12.y - finger4.y) ** 2))

                # Toggle hold state
                if length_for_hold < 0.06 and hold_state == False:
                    time.sleep(0.3)
                    hold_state = True
                elif length_for_hold < 0.06 and hold_state == True:
                    time.sleep(0.3)
                    hold_state = False

                # Move mouse if hand is close
                if length_for_move < 0.05:
                    if len(arr) >= 2:
                        xmouse += (arr[1][0] - arr[0][0]) * 0.6
                        ymouse += (arr[1][1] - arr[0][1]) * 0.6
                        if hold_state == True:
                            mouse.press(button='left')
                        else:
                            mouse.release(button='left')
                        xmouse = min(max(xmouse, 0), 1920)
                        ymouse = min(max(ymouse, 0), 1080)
                        mouse.move(xmouse, ymouse)

                # Click if necessary
                if length_for_click < 0.05:
                    mouse.click(button='left')
                    time.sleep(0.3)

    cv2.imshow('MediaPipe Hands', image)
    cv2.waitKey(4)

def MoveByHandWeb():
    """Processes frames received from the web and moves the mouse accordingly."""
    global latest_frame
    while True:
        if latest_frame is not None:
            latest_frame = cv2.rotate(latest_frame, cv2.ROTATE_90_CLOCKWISE)
            MoveByHand(latest_frame)

            latest_frame = None
        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # Web to get frames from other device, Native to use laptop camera   Web address- https://192.168.68.114:5000
    # Web / Native
    Mode = "Native"
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8,
                        min_tracking_confidence=0.5) as hands:
        arr = list()
        cap = cv2.VideoCapture(0)
        # exponential filter koef
        alpha = 0.8
        window_size = 3
        if Mode == "Native":

                # main cycle
                while cap.isOpened():
                    success, image = cap.read()

                    MoveByHand(image)
                    # Exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        elif Mode == "Web":
            threading.Thread(target=MoveByHandWeb, daemon=False).start()
            app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')

    cap.release()
    cv2.destroyAllWindows()


