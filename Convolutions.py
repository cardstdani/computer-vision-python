import cv2
import numpy as np

device = 0
video = cv2.VideoCapture(device)

blurValue = 9
kernelBlur = np.array([[1 / blurValue, 1 / blurValue, 1 / blurValue],
        [1 / blurValue, 1 / blurValue, 1 / blurValue],
        [1 / blurValue, 1 / blurValue, 1 / blurValue]] )

kernelEdge = np.array([[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]*1)

kernelEnhance = np.array([[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]*2)

while True:
    ret, frame = video.read()
    if (ret):
        #frame = cv2.flip(frame, 1)

        cv2.imshow('frame1', cv2.filter2D(frame, -1, kernelBlur))
        cv2.imshow('frame2', cv2.filter2D(frame, -1, kernelEdge))
        cv2.imshow('frame3', cv2.filter2D(frame, -1, kernelEnhance))

        cv2.imshow('frame4', cv2.Sobel(frame, -1,1,1,ksize=5))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
