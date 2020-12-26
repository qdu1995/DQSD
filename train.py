import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import keras
from keras import backend as K
import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import *

###############################################################################
def load():   
    f = h5py.File('','r')    
    f.keys()
    x = f['x'][:]
    y = f['y'][:]
    val_x = f['val_x'][:]
    val_y = f['val_y'][:]
    f.close()
    return x, y,val_x,val_y

# dimensions of our images.
img_width,  img_height =  224,224

epochs = 50
batch_size = 4

model = net(img_width,img_height)

exit()
model_checkpoint = ModelCheckpoint('D:\\.{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_weights_only=True,period=1,save_best_only=False)

train_continue = 1
if train_continue:
    model.load_weights('D:\\RGB.hdf5',by_name=True)
    model.load_weights('D:\\depthmodel.hdf5',by_name=True)
    model.load_weights('D:\\Depth.hdf5',by_name=True)
mode_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)

images,masks ,val_images,val_y= load()
#images,masks = load()
val_image = val_images[:,:,:,0:3]
print (val_image.shape)
val_deep = val_images[:,:,:,3:6]
print (val_deep.shape)


image = images[:,:,:,0:3]
print (image.shape)
deep = images[:,:,:,3:6]
print (deep.shape)

model.fit([image,deep],masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([val_image,val_deep],val_y),callbacks=[model_checkpoint,mode_lr])
