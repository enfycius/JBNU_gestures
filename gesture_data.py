import torch
import sys
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
count = 0

label = "cursor_moving"
number = 0

hand_d = pd.DataFrame(columns=[i for i in range(20)])

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
                j2 = skeleton[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
                j = j2 - j1

                j = j/np.linalg.norm(j, axis = 1)[:, np.newaxis]
        
                angle = np.arccos(np.einsum('nt,nt->n',
                    j[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],:], 
                    j[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],:]))
                
                angle = np.append(angle, label)

                # 데이터 추출 시
                df = pd.DataFrame([angle])

                # 데이터 추출 시
                hand_d = pd.concat([hand_d, df])

                print(df)
                
                # Get X
                # print(str(hand_lms.landmark[0].x))
                # Get Y
                # print(str(hand_lms.landmark[0].y))
                # Get Z
                # print(str(hand_lms.landmark[0].z))

                count = count + 1
                                    
        cv2.imshow('Webcam', image)
  
        if cv2.waitKey(1) & 0xFF == ord('q'):
          hand_d.to_csv(f"./gestures/{label + str(number)}.csv")
          break

cap.release()
cv2.destroyAllWindows()
        