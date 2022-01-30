import os
import cv2
import winsound
import faces_train
real_time_feed = cv2.VideoCapture(0)
haar_cascade = cv2.CascadeClassifier('haar_face.xml')
people=[]
for i in os.listdir(faces_train.variable()):
    people.append(i)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
while True:
    check, frame1 = real_time_feed.read()
    check, frame2 = real_time_feed.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(21,21),0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    faces_rect = haar_cascade.detectMultiScale(dilated, scaleFactor=1.2,minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)
        if confidence<100:
            #print(f'Label = {people[label]} with a confidence of {confidence}')
            cv2.putText(frame1, str(people[label]), (20, 20),cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=2)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
        else:
            winsound.PlaySound("mixkit-truck-reversing-beeps-loop-1077.wav",winsound.SND_ASYNC)
            #print('INTRUDER ALERT')
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(frame1, "INTRUDER ALLERT", (20, 20),cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=2)

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('VIDEO_FEED', frame1)
real_time_feed.release()
cv2.destroyAllWindows()