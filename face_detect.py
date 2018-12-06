import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

faceDetect=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml');
faceDetect2=cv2.CascadeClassifier('data/haarcascade_profileface.xml');

pathlist = Path('img').glob('**/*.jpg')
good = []
bad = []
cropped_path='cropped/'

for path in pathlist:
  img_path = str(path)
  test_img = cv2.imread(img_path)
  gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
  cropped_file=cropped_path+img_path.split('/')[-1]
  #detect face
  (x,y,w,h) = (0,0,0,0)
  faces = []
  faces=faceDetect.detectMultiScale(gray_img);
  if (len(faces)): (x,y,w,h)=faces[0]
  else:
    faces=faceDetect2.detectMultiScale(gray_img);
    if (len(faces)): (x,y,w,h)=faces[0]
    else:
      faces=faceDetect2.detectMultiScale(cv2.flip(gray_img,1))
      if(len(faces)): (x,y,w,h)=faces[0]
      x = gray_img.shape[1]-x-w
  if (len(faces) and w>=100): 
    good.append(img_path)
    cv2.imwrite(cropped_file, gray_img[y:y+h,x:x+w])
  else:
    bad.append(img_path) 
print(len(good), len(bad))
