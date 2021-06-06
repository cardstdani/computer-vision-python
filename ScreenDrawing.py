import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
success, i = cap.read()
width, height, c = i.shape
drawingImage = np.zeros((width, height, 3), np.uint8)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

x, y = 0, 0
color = (0, 255, 255)

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

            x2, y2 = fingerList[8][0], fingerList[8][1]
            #cv2.circle(img, (fingerList[4][0], fingerList[4][1]), 8, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, color, cv2.FILLED)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                if (x == 0) and (y == 0):
                    x, y = (x2, y2)
                cv2.line(drawingImage, (x, y), (x2, y2), color, 3)
                x, y = (x2, y2)
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.imshow("Draw", drawingImage)
    if cv2.waitKey(5) & 0xFF == 113:
        break
