#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)#webcam


while True:
    ret,img =cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'fACE',(250,210),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        roi_gray = gray[y:y+h ,x:x+w]
        roi_img = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(0,255,9),2)
            cv2.putText(img,'Eye',(300,250),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:



import pandas as pd
import numpy as np
import cv2
mbl_cascade =cv2.CascadeClassifier('cascade1.xml')
cap = cv2.VideoCapture(0)#webcam


while True:
    ret,img =cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mobiles =mbl_cascade.detectMultiScale(gray,1.3,7)
    for (x,y,w,h) in mobiles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'mobile',(250,210),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




