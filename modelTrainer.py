# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:19:45 2019

@author: Omkar Shidore
https://www.github.com/OmkarShidore
"""

import os
import cv2
import numpy as np
from PIL import Image
print('Module Versions: ')
print('OpenCv: '+cv2.__version__, '\nNumpy: '+np.__version__,'\nPillow>>Image: '+Image.__version__)

#Creatung directories
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
createFolder('./trainner/')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #empty lists for storing Lables and features
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids
faces,Ids = getImagesAndLabels('dataSet')
#training model on dataSet
recognizer.train(faces, np.array(Ids))
#trainer.yml stored in the folder trainner
recognizer.save('trainner/trainner.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


'''
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
                Id1='User Id: '+str(Id)
                Name='Name: '+keyValues[Id]
            else:
                Id="Null"
                Name="Unknown"
        else:
            Id="Unknown"
        cv2.putText(im,Id1,(x+w,y),font,0.7,(0,255,0), 2)
        cv2.putText(im,Name,(x+w,y+123),font,0.7,(0,255,0), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
'''