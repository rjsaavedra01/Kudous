from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading

app = Flask(__name__)

def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = [i.strip()[2:] for i in data if i not in garbage]
        clean = [float(i) for i in without_garbage]
        return clean
    except:
        return None  # Return None if no hand landmarks are detected

with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

translating = False
current_translation = ""
lock = threading.Lock()

def generate_frames():
    global current_translation
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_counter = 0
    process_every_n_frames = 5  # Adjust the value as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_counter % process_every_n_frames == 0:
            with lock:
                if translating:
                    data = image_processed(frame)
                    if data is not None:
                        data = np.array(data)
                        y_pred = svm.predict(data.reshape(-1, 63))
                        current_translation = str(y_pred[0])

        frame_counter += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('ndx.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_translation', methods=['POST'])
def start_translation():
    global translating
    with lock:
        translating = True
    return '', 204

@app.route('/stop_translation', methods=['POST'])
def stop_translation():
    global translating
    with lock:
        translating = False
    return '', 204

@app.route('/get_translation')
def get_translation():
    with lock:
        return jsonify({'translation': current_translation})

if __name__ == '__main__':
    app.run()
