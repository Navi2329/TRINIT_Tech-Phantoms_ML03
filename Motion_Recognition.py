'''Welcome to our CCTV Camera
Our project can be put into use where we require extreme protection
to store valuable things.
This project will detect even if there is a simple movement
and therefore this is a very efficient tool in places like museums ,jewellery shop etc.'''

import cv2,time
video = cv2.VideoCapture(0) #We are using 0 channel for webcam
first_frame = None
while True:
    check ,frame = video.read()
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY) #To inc the accuracy of the detection
    gray = cv2.GaussianBlur(gray,(21,21),0)
    #Motion is identified from a ref point. And hence ref point is reqd. So our first frame is set as ref frame.
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray)
    threshold_frame=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame=cv2.dilate(threshold_frame,None,iterations=2)
    (cntr,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cntr:
        if cv2.contourArea(contour)<1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour) #WHY?
        cv2.rectangle(frame,(x,y),(x+y,y+h),(0,255,0),3)
    cv2.imshow("cvghj",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()