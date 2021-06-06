import cv2
import numpy as np
from pyzbar.pyzbar import decode

device = 0
video = cv2.VideoCapture(device)

while True:
    s, frame = video.read()
    if (s):
        #frame = cv2.flip(frame, 1)

        for barcode in decode(frame):
            barcodeData = barcode.data.decode('utf-8')

            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 0), 5)
            pts2 = barcode.rect
            cv2.putText(frame, barcodeData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.imshow('frame1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
