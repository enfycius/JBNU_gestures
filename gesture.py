import torch
import cv2
import mediapipe as mp
import numpy as np
import threading
import struct
import wave
from pvrecorder import PvRecorder
import speech_recognition as sr
import pyperclip
import pyautogui
import time
from SSNet import SSNet

r = sr.Recognizer()
pyautogui.FAILSAFE = False

freq = 44100
duration = 5
category = ['BACK', 'CLICK', 'CURSOR_MOVING', 'DOUBLE_CLICK', 'PASTE', 'RECORDING']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SSNet()
model.to(device)
model.load_state_dict(torch.load(r"C:\JBNU_gestures\model.pt", map_location=device))
model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

s = None
thr = None
thr_copy = None
th1, th2, th3, th4 = None, None, None, None

def display_text(duration=5):
    time.sleep(duration)

def start_recording():
    recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []

    recorder.start()

    t_end = time.time() + 5
    
    while time.time() < t_end:
        frame = recorder.read()
        audio.extend(frame)

    recorder.stop()

    with wave.open(r"C:\JBNU_gestures\recording.mp3", "w") as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))

def recognize_audio():
    global s
    with sr.AudioFile(r"C:\JBNU_gestures\recording.mp3") as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        pyperclip.copy(s)

        print(s)

    except Exception as e:
        print(f"Exception: {e}")

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if detected_image.multi_hand_landmarks:
            for hand_lms in detected_image.multi_hand_landmarks:
                skeleton = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
                
                mp_drawing.draw_landmarks(image, hand_lms, 
                                          mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(255, 0, 255), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(20, 180, 90), thickness=2, circle_radius=2)
                                         )
                
                j1 = skeleton[:20]
                j2 = skeleton[1:]
                j = j2 - j1
                j = j / np.linalg.norm(j, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n', j[:-1], j[1:]))

                X = torch.tensor(angle, dtype=torch.float32).to(device)
                y_pred = model(X).argmax().item()
                action = category[y_pred]

                if action == 'CURSOR_MOVING':
                    width, height = pyautogui.size()
                    pyautogui.moveTo(int(np.mean(skeleton[:20, 0]) * width), 
                                     int(np.mean(skeleton[:20, 1]) * height), 
                                     duration=0)
                elif action == "CLICK":
                    pyautogui.click(interval=0.5)
                elif action == "DOUBLE_CLICK":
                    pyautogui.click(clicks=2, interval=0.25)
                elif action == "RECORDING":
                    if thr is None:
                        thr = threading.Thread(target=start_recording)
                        thr.start()
                elif action == "BACK":
                    pyautogui.hotkey('alt', 'left')
                elif action == "PASTE":
                    pyautogui.hotkey('ctrl', 'v')

                cv2.putText(image, action, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # if thr and not thr.is_alive():
        #     thr = None
        #     thr = threading.Thread(target=recognize_audio)
        #     thr.start()

        if thr: 
            if thr.is_alive():
                cv2.putText(image, "RECORDING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                th1 = threading.Thread(target=recognize_audio)
                th1.start()
                thr = None

        if th1:
            if th1.is_alive():
                thr_copy = threading.Thread(target=display_text)
                thr_copy.start()
            else:
                th1 = None

        if thr_copy:
            if thr_copy.is_alive() and not s is None:
                 cv2.putText(image, f"TEXT: {s}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                thr_copy = None
                s = None

        cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
