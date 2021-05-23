import cv2


def detectFaces(inputImage):
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3)

    # print(f'Number of faces found = {len(faces_rect)}')

    face = 1
    for (x, y, w, h) in faces_rect:
        # return inputImage[y:y + h, x:x + w, :] #Crop the face
        cv2.putText(inputImage, 'R ' + str(face), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(inputImage, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        face += 1
    return inputImage


device = 0
video = cv2.VideoCapture(device)

while True:
    ret, frame = video.read()
    if (ret):
        # frame = cv.flip(frame, 1)
        cv2.imshow('frame', detectFaces(frame))
        # cv.imshow('frame2', cv.Canny(detectFaces(frame), 0, 300))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()