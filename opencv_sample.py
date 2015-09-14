#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

# extract face part
cascade_file = '/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'

base_image_path = sys.argv[1]

cascade         = cv2.CascadeClassifier(cascade_file)
img = cv2.imread(base_image_path)
grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(grayscale_image, 1.3, 5)

for (x,y,w,h) in faces:
  cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.imread('yua.jpg')

# just show
#cv2.imshow('result', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# color gray
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('gray_yua.jpg', gray)


# resize
#imgsize = img.shape
#print imgsize

#simg_width = imgsize[1]/2
#simg_height = imgsize[0]/2
#simg = cv2.resize(img, (simg_width, simg_height))
#cv2.imshow('small size image', simg)
#cv2.waitKey(0)
#csv2.destroyAllWindows()

#storage = cv2.CreateMemStorage()
#img = cv2.LoadImageM('yua.jpg')
