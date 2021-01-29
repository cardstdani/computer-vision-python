import cv2
import numpy as np

device = 0
video = cv2.VideoCapture(device)

lowColor = np.array([94, 80, 2])
highColor = np.array([126, 255, 255])

tempIm = None

x1 = None
y1 = None

while True:
    ret, frame = video.read()
    if ret:
        if tempIm is None: tempIm = np.zeros(frame.shape, dtype=np.uint8)
        frame = cv2.flip(frame, 2)
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        maskColor = cv2.inRange(frameHSV, lowColor, highColor)
        maskColor = cv2.erode(maskColor, None, iterations=1)
        maskColor = cv2.dilate(maskColor, None, iterations=2)
        maskColor = cv2.medianBlur(maskColor, 13)
        cnts, _ = cv2.findContours(maskColor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for c in cnts:
            x, y2, w, h = cv2.boundingRect(c)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                x2 = (x + w // 2)
                if x1 != None:
                    tempIm = cv2.line(tempIm, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
                x1 = x2
                y1 = y2
            #cv2.rectangle(frame, (x, y2), (x + w, y2 + h), (0, 255, 0), 2)

        gray = cv2.cvtColor(tempIm, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        thInv = cv2.bitwise_not(th)
        frame = cv2.bitwise_and(frame, frame, mask=thInv)
        frame = cv2.add(frame, tempIm)

        cv2.imshow('mask', maskColor)
        cv2.imshow('frame', frame)
        cv2.imshow('frame temp', tempIm)
        # cv2.imshow('frame hsv', frameHSV)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
