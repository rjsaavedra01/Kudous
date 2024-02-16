from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
from flask_socketio import SocketIO
import speech_recognition as sr
import os

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app)

# Load the pre-trained SVM model
with open('model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Initialize variables
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

is_camera_on = True

# Function to set camera autofocus
def set_autofocus():
    # Try to set autofocus (may not work on all cameras)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # Add a small delay to allow autofocus to settle
    time.sleep(2)

# Set autofocus if available
set_autofocus()

predicted_sentence = ""
last_prediction_time = time.time()
last_letter_append_time = time.time()

def process_hand_image(hand_img, is_camera_on):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    if is_camera_on:  # Only process hand gestures if the camera is on
        try:
            hand_landmarks = output.multi_hand_landmarks[0]
            hand_confidence = output.multi_handedness[0].classification[0].score
            landmarks_data = [float(i.strip()[2:]) for i in str(hand_landmarks).strip().split('\n') if
                              i not in ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']]
            return np.array(landmarks_data), hand_confidence
        except:
            return np.zeros(63, dtype=int), 0.0
    else:
        return np.zeros(63, dtype=int), 0.0

def generate_frames(is_camera_on):
    global predicted_sentence
    global last_prediction_time
    global last_letter_append_time

    cap.set(3, 640)  # Set width to 640
    cap.set(4, 480)  # Set height to 480

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hand_data, hand_confidence = process_hand_image(frame, is_camera_on)

        if hand_confidence >= 0.5:  # Adjust this threshold based on your preferences
            if np.all(hand_data == 0):
                if time.time() - last_prediction_time >= 59:
                    predicted_sentence = ""
            else:
                last_prediction_time = time.time()

                distance_landmarks = np.linalg.norm(hand_data[0:3] - hand_data[16:19])
                distance_text = f"Distance: {distance_landmarks:.2f} pixels"
                cv2.putText(frame, distance_text, (int(frame.shape[1] / 4), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

                y_pred = svm_model.predict(hand_data.reshape(1, -1))

                if time.time() - last_letter_append_time >= 2:
                    predicted_sentence += str(y_pred[0])
                    last_letter_append_time = time.time()

        # Display the predicted sentence in the lower part of the frame
        cv2.putText(frame, f"Predicted Text: {predicted_sentence}", (50, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def play_video(filename, text):
    with open(filename, 'rb') as video_file:
        video_data = video_file.read()
        socketio.emit('play_video', {'video_data': video_data})
        socketio.emit('update_text', {'text': text})

def listen_to_microphone():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source, timeout=1)
                text = recognizer.recognize_google(audio)
                print("You said:", text)

                # Update the convertedText element with the recognized text
                socketio.emit('update_text', {'text': text})

                video_folder = os.path.join(os.getcwd(), "mp4")
                video_file = os.path.join(video_folder, f"{text.upper()}.mp4")

                if os.path.exists(video_file):
                    print(f"Playing video for user input: {text}")
                    play_video(video_file, text)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Error connecting to Google Speech Recognition service: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asl_to_text', methods=['GET', 'POST'])
def asl_to_text():
    if request.method == 'POST':
        # Handle any post request if needed
        pass
    return render_template('ASL-to-text.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(is_camera_on), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predicted_text')
def get_predicted_text():
    global predicted_sentence
    return jsonify({'predicted_text': predicted_sentence})

@app.route('/clear_predicted_text', methods=['POST'])
def clear_predicted_text():
    global predicted_sentence
    predicted_sentence = ""
    return 'Predicted text cleared'

@app.route('/release_camera', methods=['POST'])
def release_camera():
    global cap
    cap.release()
    return 'Camera released'

@app.route('/delete_char', methods=['POST'])
def delete_char():
    global predicted_sentence
    if predicted_sentence:
        predicted_sentence = predicted_sentence[:-1]  # Remove the last character
    return jsonify({'predicted_text': predicted_sentence})

@app.route('/get-started', methods=['GET', 'POST'])
def get_started():
    if request.method == 'POST':
        # Handle any post request if needed
        pass
    return render_template('get-started.html')

# New route to handle form submission
@app.route('/submit_text', methods=['POST'])
def submit_text():
    user_input = request.form.get('userInput')
    video_folder = os.path.join(os.getcwd(), "mp4")
    video_file = os.path.join(video_folder, f"{user_input.upper()}.mp4")

    if os.path.exists(video_file):
        print(f"Playing video for user input: {user_input}")
        play_video(video_file, user_input)

    return render_template('index.html')  # Redirect to the main page after form submission

if __name__ == '__main__':
    threading.Thread(target=listen_to_microphone).start()
    socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
