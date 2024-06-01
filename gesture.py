import torch
import sys
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from SSNet import SSNet
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time
import pyautogui
import threading
import struct
import wave
from pvrecorder import PvRecorder
import speech_recognition as sr
import pyperclip

r = sr.Recognizer()

pyautogui.FAILSAFE = False

freq = 44100
duration = 5

category = ['BACK', 'CLICK', 'CURSOR_MOVING', 'DOUBLE_CLICK']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SSNet()

model.to(device)

model.load_state_dict(torch.load("./best_model.pt", map_location=device))

model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
count = 0

label = "back"
number = 3

s = None

thr = None
thr_copy = None

hand_d = pd.DataFrame(columns=[i for i in range(20)])

def display_text():
    t_end = time.time() + 5
    
    while time.time() < t_end:
        pass

def start_recording():
    recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []

    recorder.start()

    t_end = time.time() + 5
    
    while time.time() < t_end:
        frame = recorder.read()
        audio.extend(frame)

    recorder.stop()

    with wave.open("./recording.mp3", "w") as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if detected_image.multi_hand_landmarks:
            for hand_lms in detected_image.multi_hand_landmarks:
                skeleton = np.zeros((21, 3))
    
                mp_drawing.draw_landmarks(image, hand_lms, 
                                          mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                            color=(255, 0, 255), thickness=4, circle_radius=2),
                                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                            color=(20, 180, 90), thickness=2, circle_radius=2)
                                        )
                
                for idx, data in enumerate(hand_lms.landmark):
                    skeleton[idx] = [data.x, data.y, data.z]

                j1 = skeleton[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],:]

                print("x:", np.mean(skeleton[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 0]))
                x = np.mean(skeleton[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 0])
                print("y:", np.mean(skeleton[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 1]))
                y = np.mean(skeleton[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 1])

                j2 = skeleton[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
                j = j2 - j1

                j = j/np.linalg.norm(j, axis = 1)[:, np.newaxis]
        
                angle = np.arccos(np.einsum('nt,nt->n',
                    j[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],:], 
                    j[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],:]))

                df = pd.DataFrame([angle])
                # 데이터 추출 시
                # df = pd.DataFrame([angle, label])
                hand_d = pd.concat([hand_d, df])

                X = torch.from_numpy(angle)
                X = X.type(torch.FloatTensor)
                X.to(device)

                y_pred = model(X)
                
                y_pred = torch.argmax(y_pred, -1)

                print(y_pred.item())

                if category[int(y_pred.item())] == 'CURSOR_MOVING':
                    old_x, old_y = pyautogui.position()
                    width, height = pyautogui.size()
                    current_x = int(x * width)
                    current_y = int(y * height)
                    pyautogui.moveTo(current_x, current_y, duration=0)
                elif category[int(y_pred.item())] == "CLICK":
                    pyautogui.click()
                elif category[int(y_pred.item())] == "DOUBLE_CLICK":
                    pyautogui.click(clicks=2, interval=0.25)
                elif category[int(y_pred.item())] == "BACK":
                    if thr == None:
                        thr = threading.Thread(target=start_recording)
                        thr.start()

                image = cv2.putText(image, category[int(y_pred.item())], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # 데이터 추출 시
                # hand_d = pd.concat([hand_d, df])
                
                # Get X
                # print(str(hand_lms.landmark[0].x))
                # Get Y
                # print(str(hand_lms.landmark[0].y))
                # Get Z
                # print(str(hand_lms.landmark[0].z))

                # print(hand_d)
                # print(df)
                count = count + 1
                
        
        try:
            if thr is not None:
                if thr.is_alive():
                    image = cv2.putText(image, "RECORDING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    text_recording=sr.AudioFile('./recording.mp3')

                    with text_recording as source:
                        audio = r.record(source)
                    try:
                        s = r.recognize_google(audio)
                        print("Text: "+s)

                        pyperclip.copy(s)

                        thr_copy = threading.Thread(target=display_text)
                        thr_copy.start()
                    except Exception as e:
                        print("Exception: "+str(e))
                    thr = None
        except Exception as e:
            pass

        try:
            if thr_copy is not None:
                if thr_copy.is_alive():
                    image = cv2.putText(image, f"TEXT: {s}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    thr_copy = None
                    s = None
        except Exception as e:
            pass

        cv2.imshow('Webcam', image)
  
        if cv2.waitKey(1) & 0xFF == ord('q'):
          hand_d.to_csv(f"./gestures/{label + str(number)}.csv")
          break

cap.release()
cv2.destroyAllWindows()
        