import  numpy  as  np
import cv2
import skimage
import csv
import pandas as pd
from pandas import DataFrame as df
from CoSaMP import cosamp
from time import time

tstart = time()
Sprasity = 15000
epsilon=1e-10



#Image initialization
image = cv2.imread('lib.jpg',0)
image = cv2.resize(image,None, fx=0.5, fy=0.5)
#image = cv2.resize(image, (391 , 256 ))
image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
n_rows, n_cols=image.shape
noisy_image = skimage.img_as_ubyte(skimage.util.random_noise(image, mode='s&p', seed=None, clip=True, amount= 0.3))
print (image.shape, 'Image shape')

flat_img =  np.asarray(noisy_image).reshape(-1)
print (flat_img.shape[0])
#load Dic
#with open('compactdict.csv') as f:
#    reader = csv.reader(f)
#    Dict = []
#   for row in reader:
#        Dict.extend(row)
asize = flat_img.shape[0]
A = np.random.normal(0, 1, [n_rows,asize])
print (A.shape, 'Phi (sampling matrix) shape')
y=np.dot(A,flat_img)
print (y.shape,'measurements vector shape')
#cv2.imshow('Y', y)
#cv2.waitKey()



#Dict= pd.read_csv('D:\Google Drive\Code\cs-object-detection-v2\compactdict.csv')
#Dict.head()
#print type(Dict)
#print Dict.shape

#Corro = df.corrwith( y, Dict, axis=1, drop=True)
#print Corro
#load Dic
##with open('compactdict.csv') as f:
#  reader = csv.reader(f)
#  Dict = []
#   for row in reader:
#        Dict.extend(row)
#reader = csv.reader(open("compactdict2.csv", "rb"), delimiter=",")
#x = list(reader)
#Dict = np.array(x).astype("double")/255
#Dict2 = Dict.transpose((1, 0))
#print (Dict.shape, 'Dictionary shape')
#CorrMat = []
#Num_Frame = Dict.shape[1]-1

#for itt in  range(0,Num_Frame) :
#   vect= Dict[:,itt]
#   print vect.shape, 'vaect'
#   coro = np.corrcoef(vect, samp)
#   print coro [1]
#   CorrMat.append(coro [1])
print (n_cols, n_rows)
#testdict =Dict[:,150]
#print testdict.shape
#testdict =testdict.reshape(n_rows, n_cols)
#testdict = testdict.transpose((1, 0))
#print testdict.shape, 'dict after transpose '

s= 2000
#Sprasity
phi=A
print (phi.shape, 'phi ')
u=y
x_est = cosamp(phi,y, s,epsilon)
print (x_est.shape)
x_est = x_est.reshape(n_cols,n_rows)
#x_est = cv2.resize(x_est, (976,640))
print ('Approximation Done')
cv2.imshow('clean image', image)
cv2.waitKey()
cv2.imshow('noisy image', noisy_image)
cv2.waitKey()
cv2.imshow('X_EST', x_est)
cv2.waitKey()
#cv2.imwrite("est_{}sparse_{}epsilon_{}.png".format(int (s), float (epsilon),it), x_est)

tstop = time() - tstart
print ('elapsed time =', tstop)
