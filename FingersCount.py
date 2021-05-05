import math

import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
success, i = cap.read()
width, height, c = i.shape

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

x, y = 0, 0
color = (0, 255, 255)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Por cada mano
            fingerList = []
            for id, lm in enumerate(handLms.landmark):  # Por cada dedo detectado
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                fingerList.append([cx, cy])

            fingersUp = 0

            if fingerList[tipIds[0]][1] > fingerList[tipIds[0] - 1][1]:
                fingersUp += 1

            for id in range(1, 5):
                if fingerList[tipIds[id]][1] < fingerList[tipIds[id] - 2][1]:
                    fingersUp += 1

            x2, y2 = fingerList[8][0], fingerList[8][1]
            x, y = fingerList[4][0], fingerList[4][1]
            cv2.circle(img, (x, y), 8, color, cv2.FILLED)
            cv2.putText(img, str(fingersUp), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == 113:
        break
