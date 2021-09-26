import numpy as np
import cv2

img = cv2.imread('shaperec2.png',0)

#create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

img2 = cv2.imread('noised.png',0)
#create a CLAHE object (Arguments are optional).
clahe2= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl12 = clahe2.apply(img2)


cv2.imshow('first image',img)
cv2.imwrite('clahe_1.jpg',cl1)
cv2.imshow('local histogram',cl1)
cv2.waitKey()
cv2.imshow('second image',img2)
cv2.imwrite('clahe_2.jpg',cl12)
cv2.imshow('local histogram',cl12)
cv2.waitKey()
