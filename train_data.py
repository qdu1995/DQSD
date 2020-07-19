import cv2
import numpy as np
#import cPickle as pickle
import h5py
import random
import operator
PIC_PATH = ''
SALMAP_PATH = ''
deep_path1 = '';

VAL_PATH = ''
VALMASK_PATH = ''
deep_val = '';

datalist = open(PIC_PATH+'list.txt','r')
namelist=[l.strip('\n') for l in datalist.readlines()]

sallist = open(SALMAP_PATH+'list.txt','r')
sallist=[l.strip('\n') for l in sallist.readlines()]

val_datalist = open(VAL_PATH+'list.txt','r')
val_namelist=[l.strip('\n') for l in val_datalist.readlines()]

val_sallist = open(VALMASK_PATH+'list.txt','r')
val_sallist=[l.strip('\n') for l in val_sallist.readlines()]

depthlist = open(deep_path1+'list.txt','r')
depthnamelist=[l.strip('\n') for l in depthlist.readlines()]

depthlist_val = open(deep_val+'list.txt','r')
depthnamelist_val=[l.strip('\n') for l in depthlist_val.readlines()]

size=224
NumSample=len(namelist)
val_num = len(val_namelist)

X1 = np.zeros((NumSample,size, size,3), dtype='float32')
d1 = np.zeros((NumSample,size, size), dtype='float32')
X2 = np.zeros((NumSample,size, size,6), dtype='float32')
Y1 = np.zeros((NumSample,size,size,1), dtype='uint8')
#Y1 = np.zeros((NumSample,output_h,output_w,1), dtype='float32')
NumAll = NumSample
print (NumAll)

print (val_num)
VAL_X = np.zeros((val_num,size, size,3), dtype='float32')
VAL_d1 = np.zeros((val_num,size, size), dtype='float32')
VAL_X2 = np.zeros((val_num,size, size,6), dtype='float32')
VAL_Y = np.zeros((val_num,size,size,1), dtype='uint8')
 
for i in range(NumSample):
    img = cv2.imread(PIC_PATH+namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
    #depth
    dep = cv2.imread(deep_path1+depthnamelist[i], cv2.IMREAD_GRAYSCALE)
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

    label = cv2.imread(SALMAP_PATH+sallist[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(size,size),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    Y1[i]=label.reshape(size,size,1)

for i in range(val_num):
    img = cv2.imread(VAL_PATH+val_namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)/255.
    VAL_X[i]=img
         
    valdep = cv2.imread(deep_val+depthnamelist_val[i], cv2.IMREAD_GRAYSCALE)
    valdep = cv2.resize(valdep,(size,size),interpolation=cv2.INTER_CUBIC)
    valdep=valdep.astype(np.float32)/255.
    VAL_d1[i]=valdep
    VAL_X2[i,:,:,0]=VAL_X[i,:,:,0]
    VAL_X2[i,:,:,1]=VAL_X[i,:,:,1]
    VAL_X2[i,:,:,2]=VAL_X[i,:,:,2]
    VAL_X2[i,:,:,3]=VAL_d1[i,:,:]
    VAL_X2[i,:,:,4]=VAL_d1[i,:,:]
    VAL_X2[i,:,:,5]=VAL_d1[i,:,:]
     
    label = cv2.imread(VALMASK_PATH+val_sallist[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(size,size),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    VAL_Y[i]=label.reshape(size,size,1)

f = h5py.File('','w') 

f['x'] = X2      
f['y'] = Y1
f['val_x'] = VAL_X2
f['val_y'] = VAL_Y
f.close()        