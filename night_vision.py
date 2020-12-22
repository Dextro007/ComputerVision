import numpy as np
import cv2
cap = cv2.VideoCapture('vid1_IR.avi')
while(cap.isOpened()):
    ret, img = cap.read()
    # cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    # cv2.imshow('video', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((10, 10), np.float32)/100
    smooth = cv2.filter2D(gray, -1, kernel)
    retval, thresh  = cv2.threshold(smooth, 50, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
       rect = cv2.minAreaRect(cnt)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       img = cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    #cv2.imshow('gray', thresh)
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
