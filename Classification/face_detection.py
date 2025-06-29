import cv2 as cv
import numpy as np

image=cv.imread("Designer.jpg",1)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


faces=face_cascade.detectMultiScale(image,1.2,5)

for (x,y,w,h) in faces:
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,255),4)
    cv.putText(image,'Face Detected',(x,y+h+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,25), 1, cv.LINE_AA) 

cv.imshow("image",image)
cv.waitKey(0)
cv.destroyAllWindows()  