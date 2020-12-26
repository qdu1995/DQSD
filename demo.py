import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from model import *
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
###############################################################################
def load():    
    f = h5py.File('','r')   
    f.keys()
    X = f['x'][:]
    f.close()
    return X

images = load()
image = images[:,:,:,0:3]
print (image.shape)
deep = images[:,:,:,3:6]
print (deep.shape)

img_width,  img_height = 224,224 

model = net(img_width,img_height)
model.load_weights('C:net.hdf5',by_name=False)

img_pre=model.predict([image,deep],batch_size=4, verbose=1)
for i in range(img_pre.shape[0]):
    img = img_pre[i]
    img = array_to_img(img)
    img.save("result\\%d.jpg"%(1+i))
    
#print(img_pre[i])



