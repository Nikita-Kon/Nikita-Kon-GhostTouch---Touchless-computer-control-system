from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

latest_frame = None

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

def display_frames():
    global latest_frame

    while True:
        if latest_frame is not None:
            cv2.imshow('Received Frame', latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    import threading

    threading.Thread(target=display_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
