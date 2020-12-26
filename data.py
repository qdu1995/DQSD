import cv2
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt


RGB_path = 'E\\RGB\\'#path
deep_path = 'E:\\depth\\';#path

datalist = open(RGB_path+'list.txt','r')
namelist=[l.strip('\n') for l in datalist.readlines()]
depthlist = open(deep_path+'list.txt','r')
depthnamelist=[l.strip('\n') for l in depthlist.readlines()]

size=224
NumSample=len(namelist)

X1 = np.zeros((NumSample,size, size,3), dtype='float32')
d1 = np.zeros((NumSample,size, size), dtype='float32')
X2 = np.zeros((NumSample,size, size,6), dtype='float32')

NumAll = NumSample
print (NumAll)

for i in range(NumSample):
    #RGB
    img = cv2.imread(RGB_path+namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
    #depth
    dep = cv2.imread(deep_path+depthnamelist[i], cv2.IMREAD_GRAYSCALE)
    dep = cv2.resize(dep,(size,size),interpolation=cv2.INTER_CUBIC)
    
    img=img.astype(np.float32)/255.
    dep=dep.astype(np.float32)/255.
    if( (img.shape>(size,size,3))-(img.shape<(size,size,3)) == 0):
        X1[i]=img
    else:
        print ('error')
    d1[i]=dep
    X2[i,:,:,0]=X1[i,:,:,0]
    X2[i,:,:,1]=X1[i,:,:,1]
    X2[i,:,:,2]=X1[i,:,:,2]
    X2[i,:,:,3]=d1[i,:,:]
    X2[i,:,:,4]=d1[i,:,:]
    X2[i,:,:,5]=d1[i,:,:]


f = h5py.File('E:\\.h5','w')#path 
f['x'] = X2      
f.close()