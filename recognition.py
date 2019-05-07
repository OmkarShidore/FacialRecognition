# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:37:50 2019

@author: Omkar Shidore
https://www.github.com/OmkarShidore
"""
import os
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

def keyValuePairs(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    Ids=[]
    UserNames=[]
    for imagePath in imagePaths:   
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        UserName=str(os.path.split(imagePath)[-1].split(".")[2])    
        Ids.append(Id)
        UserNames.append(UserName)
    keyValues = dict(zip(Ids, UserNames))
    return Ids,UserNames,keyValues
Ids,UserNames,keyValues=keyValuePairs('dataSet')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf>50):
            if Id in keyValues.keys():
                Id1= 'User Id  : '+str(Id)
                Name='User Name: '+keyValues[Id]
            else:
                Id="Null"
                Name="Unknown"
        else:
            Id="Unknown"
        cv2.putText(im,Id1,(x,y+h+20),font,0.7,(0,255,0), 2)
        cv2.putText(im,Name,(x,y+h+38),font,0.7,(0,255,0), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()

#(124,185,234)
'''
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf>50):
            if(Id==1):
                Id="Anirban"
            elif(Id==2):
                Id="Sam"
        else:
            Id="Unknown"
        cv2.putText(im,Id,(x,y+h),font,2,(0,255,0), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
'''