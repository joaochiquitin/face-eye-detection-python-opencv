import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/Joao/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('C:/Users/Joao/Downloads/opencv/sources/data/haarcascades\haarcascade_eye.xml')

#Inicia a WebCam (câmera).
VideoCapture = cv2.VideoCapture(0)

while(True):
    ret, img = VideoCapture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detecção da Face (rosto).
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        #Detecção dos olhos
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img', img)

    #Caso aperte a tecla F ele fecha.
    if cv2.waitKey(1) & 0xff == ord('f'):
        break

VideoCapture.release()
cv2.destroyAllWindows()