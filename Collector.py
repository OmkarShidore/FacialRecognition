# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:19:15 2019

@author: Omkar Shidore
https://www.github.com/OmkarShidore
"""
import cv2
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
#created folder for saving dataset images of user
createFolder('./dataSet/')

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Id=input('Enter your id: ')
UserName=input('Enter Name: ')
sampleNum=0
while True:
    ret, img = cam.read()
    ret, img1 = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        sampleNum1=str(sampleNum)
        text=sampleNum
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ UserName+'.'+str(sampleNum)+'.'+ ".jpg", gray[y:y+h,x:x+w])
        
        cv2.putText(img1,sampleNum1, (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255), 3)
        cv2.imshow('Scans',img1)
        
    if cv2.waitKey(1)==27:
        break
    # break if the sample number is morethan 20
    elif sampleNum>100:
        break
cam.release()
cv2.destroyAllWindows()