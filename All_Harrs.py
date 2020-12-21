# All Haar Detections

import cv2


def Pedestrian():
    
    video_src = 'pedestrians.avi'

    cap = cv2.VideoCapture(video_src)

    bike_cascade = cv2.CascadeClassifier('pedestrian.xml')

    while True:
        ret, img = cap.read()
            
        
        if (type(img) == type(None)):
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bike = bike_cascade.detectMultiScale(gray,1.3,2)

        for(a,b,c,d) in bike:
            cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
        
        cv2.imshow('video', img)
        
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


def Car():
    
    video_src = 'cars2.avi'

    cap = cv2.VideoCapture(video_src)

    car_cascade = cv2.CascadeClassifier('cars.xml')


    while True:
        ret, img = cap.read()
       
        if (type(img) == type(None)):
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.3, 2)


        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        
        cv2.imshow('video', img)
       
        
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()
    

def Bike():

    cascade_src = 'two_wheeler.xml'

    video_src = 'bikes.mp4'

    cap = cv2.VideoCapture(video_src)

    car_cascade = cv2.CascadeClassifier(cascade_src)


    while True:
        ret, img = cap.read()
        
        if (type(img) == type(None)):
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cars = car_cascade.detectMultiScale(gray,1.01, 1)


        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        
        cv2.imshow('video', img)
        
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()



def Bus():

    cascade_src = 'Bus_front.xml'

    video_src = 'bus1.mp4'

    cap = cv2.VideoCapture(video_src)

    car_cascade = cv2.CascadeClassifier(cascade_src)


    while True:
        ret, img = cap.read()
        
        if (type(img) == type(None)):
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cars = car_cascade.detectMultiScale(gray,1.01, 1)


        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        
        cv2.imshow('video', img)
        
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


