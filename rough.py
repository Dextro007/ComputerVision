import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_eye.xml')
#upperbody_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_upperbody.xml')
#lowerbody_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_lowerbody.xml')
#fullbody_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_fullbody.xml')
#car_cascade = cv2.CascadeClassifier('G:\\Programs\\PyPrograms\\haarcascade_car.xml')

cap = cv2.VideoCapture("videoplayback.mp4")

while (cap.isOpened()):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2,4)
    #fullbody = fullbody_cascade.detectMultiScale(gray,1.05,3)
    #car = car_cascade.detectMultiScale(gray,1.1,5)
    #for(cx,cy,cw,ch) in car:
     #   cv2.rectangle(img,(cx,cy), (cx+cw, cy+ch), (100,0, 100), 2)
    #for (fx,fy,fw,fh) in fullbody:
     #   cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (120, 55, 200), 2)
     #   roi_c = gray[fy:fy + fh, fx:fx + fw]
      #  roi_g = img[fy:fy + fh, fx:fx + fw]
       # upperbody = upperbody_cascade.detectMultiScale(roi_g,1.3,3)
     #   lowerbody = lowerbody_cascade.detectMultiScale(roi_g,1.3,3)
      #  for(ux, uy, uw, uh) in upperbody:
            #cv2.rectangle(roi_c, (ux,uy), (ux+uw, uy+uh), (120,150,130),2)
     #   for(lx,ly,lw,lh) in lowerbody:
     #       cv2.rectangle(roi_c, (lx,ly), (lx+lw, ly+lh), (0,0,255),2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 3)
        #for (ex, ey, ew, eh) in eyes:
         #   cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
