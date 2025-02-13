# Import libraries
import time
import cv2
import math
import mediapipe as mp
import mouse

from queue import Queue

# solution hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# gesture recognise options
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# global result variables
recognized_gesture = ""
gesture_score = 0.0
hold_state = False

def smooth(prev, curr):
    return alpha * curr + (1 - alpha) * prev

def moving_average(prev_values, curr_value):
    if len(prev_values) == window_size:
        prev_values.pop(0)
    prev_values.append(curr_value)
    return sum(prev_values) / len(prev_values)
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognized_gesture, gesture_score
    if result.gestures:
        gesture = result.gestures[0][0]
        recognized_gesture = gesture.category_name
        gesture_score = gesture.score * 100

# Gesture network options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=r"C:\Users\Aser\PycharmProjects\handDrawing\gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


with GestureRecognizer.create_from_options(options) as recognizer, \
        mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.5) as hands:
    arr = list()
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    arr_counter = 0
    # exponential filter koef
    alpha = 0.8
    window_size = 3
    # main cycle
    while cap.isOpened():
        success, image = cap.read()
        # horizontal image flip
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue
        # Gesture recognition
        # if frame_timestamp_ms % 8 == 0:
        #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        #     recognizer.recognize_async(mp_image, frame_timestamp_ms)
        # frame_timestamp_ms += 1

        # Hand landmark detection, draw
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if hand_landmarks.landmark:

                    finger8_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    finger4_point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    cursor_landmarkX = (finger8_point.x + finger4_point.x) / 2
                    cursor_landmarkY = (finger4_point.y + finger8_point.y) / 2

                    # get landmark coordinates
                    xhand = int((cursor_landmarkX * 640))
                    yhand = int(cursor_landmarkY * 480)

                    xhand = min(max(xhand, 100), 540)
                    yhand = min(max(yhand, 85), 395)

                    def map_rangeX(value):
                        return (value - 100) * (1920 - 0) / (540 - 100) + 0
                    def map_rangeY(value):
                        return (value - 85) * (1080 - 0) / (395 - 85) + 0

                    xhand = map_rangeX(xhand)
                    yhand = map_rangeY(yhand)
                    # add line coordinates
                    cord_mouse = mouse.get_position()
                    xmouse = cord_mouse[0]
                    ymouse = cord_mouse[1]

                    # put contur for text on image
                    cv2.putText(image, f"{recognized_gesture}, Score: {gesture_score:.2f}%", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 4)
                    # put text on image
                    cv2.putText(image, f"{recognized_gesture}, Score: {gesture_score:.2f}%", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.line(image, (100, 85), (540, 85), (0, 255, 0), 1)
                    cv2.line(image, (540, 85), (540, 395), (0, 255, 0), 1)
                    cv2.line(image, (100, 85), (100, 395), (0, 255, 0), 1)
                    cv2.line(image, (100, 395), (540, 395), (0, 255, 0), 1)
                    # show image

                    prev_x_values = []
                    prev_y_values = []

                    xhand = moving_average(prev_x_values, xhand)
                    yhand = moving_average(prev_y_values, yhand)
                # update x y
                if len(arr) == 0 or len(arr) == 1:
                    arr.append([xhand, yhand])
                else:
                    smoothed_x = smooth(arr[0][0], arr[1][0])
                    smoothed_y = smooth(arr[0][1], arr[1][1])
                    arr[0] = [smoothed_x, smoothed_y]
                    arr[1] = [xhand, yhand]  # Assign a new list to arr[1]
                # exponential filter for arr(x, y)

                finger8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                finger5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                finger12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                length_for_move = math.sqrt(((finger8.x - finger4.x) ** 2) + ((finger8.y - finger4.y) ** 2))
                length_for_click = math.sqrt(((finger4.x - finger5.x) ** 2) + ((finger4.y - finger5.y) ** 2))
                length_for_hold = math.sqrt(((finger12.x - finger4.x) ** 2) + ((finger12.y - finger4.y) ** 2))

                if length_for_hold < 0.06 and hold_state == False:
                    time.sleep(0.3)
                    hold_state = True
                elif length_for_hold < 0.06 and hold_state == True:
                    time.sleep(0.3)
                    hold_state = False

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

                # 0.06
                if length_for_click < 0.05:
                    mouse.click(button='left')
                    time.sleep(0.3)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(4) & 0xFF == 27:
            break
cap.release()



