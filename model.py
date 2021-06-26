import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
#import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import pickle as pickle

import itertools
def net(img_width,img_height):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width,3)
    print (K.image_data_format())

    inputs=Input(input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)
    
    maxpool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(maxpool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    maxpool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), padding='same',  name='block3_conv1')(maxpool2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    conv3_4 = Conv2D(256, (3, 3), padding='same', name='block3_conv4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    conv3_4 = Activation('relu')(conv3_4)
    maxpool3 = MaxPooling2D((2, 2), name='block3_pool')(conv3_4)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), padding='same',  name='block4_conv1')(maxpool3)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_3 = Activation('relu')(conv4_3)
    conv4_4 = Conv2D(512, (3, 3), padding='same', name='block4_conv4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    conv4_4 = Activation('relu')(conv4_4)
    maxpool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),   padding='same', name='block5_conv1')(maxpool4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3),   padding='same', name='block5_conv2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Activation('relu')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3),   padding='same', name='block5_conv3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_3 = Activation('relu')(conv5_3)
###############################################################################
    input_depth = Input((img_height, img_width,3))
    # Block a
    conva_1 = Conv2D(64, (3, 3), padding='same',name='block_a1')(input_depth)
    conva_1 = BatchNormalization(name='batch_a1')(conva_1)
    conva_1 = Activation('relu')(conva_1)
    conva_2 = Conv2D(64, (3, 3), padding='same', name='block_a2')(conva_1)
    conva_2 = BatchNormalization(name='batch_a2')(conva_2)
    conva_2 = Activation('relu')(conva_2)
    maxpoola = MaxPooling2D((2, 2), strides=(2, 2))(conva_2)

    # Block b
    convb_1 = Conv2D(128, (3, 3), padding='same',name='block_b1')(maxpoola)
    convb_1 = BatchNormalization(name='batch_b1')(convb_1)
    convb_1 = Activation('relu')(convb_1)
    convb_2 = Conv2D(128, (3, 3), padding='same', name='block_b2')(convb_1)
    convb_2 = BatchNormalization(name='batch_b2')(convb_2)
    convb_2 = Activation('relu')(convb_2)
    maxpoolb = MaxPooling2D((2, 2), strides=(2, 2))(convb_2)

    # Block c
    convc_1 = Conv2D(256, (3, 3), padding='same',  name='block_c1')(maxpoolb)
    convc_1 = BatchNormalization(name='batch_c1')(convc_1)
    convc_1 = Activation('relu')(convc_1)
    convc_2 = Conv2D(256, (3, 3), padding='same', name='block_c2')(convc_1)
    convc_2 = BatchNormalization(name='batch_c2')(convc_2)
    convc_2 = Activation('relu')(convc_2)
    convc_3 = Conv2D(256, (3, 3), padding='same', name='block_c3')(convc_2)
    convc_3 = BatchNormalization(name='batch_c3')(convc_3)
    convc_3 = Activation('relu')(convc_3)
    convc_4 = Conv2D(256, (3, 3), padding='same', name='block_c4')(convc_3)
    convc_4 = BatchNormalization(name='batch_c4')(convc_4)
    convc_4 = Activation('relu')(convc_4)
    maxpoolc = MaxPooling2D((2, 2),strides=(2, 2))(convc_4)

    # Block d
    convd_1 = Conv2D(512, (3, 3), padding='same', name='block_d1')(maxpoolc)
    convd_1 = BatchNormalization(name='batch_d1')(convd_1)
    convd_1 = Activation('relu')(convd_1)
    convd_2 = Conv2D(512, (3, 3), padding='same', name='block_d2')(convd_1)
    convd_2 = BatchNormalization(name='batch_d2')(convd_2)
    convd_2 = Activation('relu')(convd_2)
    convd_3 = Conv2D(512, (3, 3), padding='same', name='block_d3')(convd_2)
    convd_3 = BatchNormalization(name='batch_d3')(convd_3)
    convd_3 = Activation('relu')(convd_3)
    convd_4 = Conv2D(512, (3, 3), padding='same', name='block_d4')(convd_3)
    convd_4 = BatchNormalization(name='batch_d4')(convd_4)
    convd_4 = Activation('relu')(convd_4)
    maxpoold = MaxPooling2D((2, 2), strides=(2, 2))(convd_4)

    # Block 5
    conve_1 = Conv2D(512, (3, 3),   padding='same',name='block_e1')(maxpoold)
    conve_1 = BatchNormalization(name='batch_e1')(conve_1)
    conve_1 = Activation('relu')(conve_1)
    conve_2 = Conv2D(512, (3, 3),   padding='same', name='block_e2')(conve_1)
    conve_2 = BatchNormalization(name='batch_e2')(conve_2)
    conve_2 = Activation('relu')(conve_2)
    conve_3 = Conv2D(512, (3, 3),   padding='same', name='block_e3')(conve_2)
    conve_3 = BatchNormalization(name='batch_e3')(conve_3)
    conve_3 = Activation('relu')(conve_3)

    #de conv begin f
    convf_1 = Conv2D(512, (3, 3),   padding='same', name='block_f1')(conve_3)
    convf_1 = BatchNormalization(name='batch_f1')(convf_1)
    convf_1 = Activation('relu')(convf_1)
    uppoolf=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convf_1)

    #cross 1 g
    g = concatenate([convd_4,uppoolf],axis=-1)
    convg_1 = Conv2D(512, (3, 3),   padding='same', name='block_g1')(g)
    convg_1 = BatchNormalization(name='batch_g1')(convg_1)
    convg_1 = Activation('relu')(convg_1)
    convg_2 = Conv2D(512, (3, 3),   padding='same', name='block_g2')(convg_1)
    convg_2 = BatchNormalization(name='batch_g2')(convg_2)
    convg_2 = Activation('relu')(convg_2)
    
    uppoolg=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convg_2)
    convg_4 = Conv2D(64, (3, 3),   padding='same', name='block_g4',kernel_initializer='TruncatedNormal')(convg_2)
    convg_4 = BatchNormalization(name='batch_g4')(convg_4)
    convg_4 = Activation('relu')(convg_4)
    
    #cross 2 h
    h = concatenate([convc_4,uppoolg],axis=-1)
    convh_1 = Conv2D(256, (3, 3),   padding='same',name='block_h1')(h)
    convh_1 = BatchNormalization(name='batch_h1')(convh_1)
    convh_1 = Activation('relu')(convh_1)
    convh_2 = Conv2D(256, (3, 3),   padding='same', name='block_h2')(convh_1)
    convh_2 = BatchNormalization(name='batch_h2')(convh_2)
    convh_2 = Activation('relu')(convh_2)
    
    uppoolh=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convh_2)  
    convh_3 = Conv2D(64, (3, 3),   padding='same', name='block_h3',kernel_initializer='TruncatedNormal')(convh_2)
    convh_3 = BatchNormalization(name='batch_h3')(convh_3)
    convh_3 = Activation('relu')(convh_3)
    
    #cross 3
    i = concatenate([convb_2,uppoolh],axis=-1)
    convi_1 = Conv2D(128, (3, 3),   padding='same', name='block_i1')(i)
    convi_1 = BatchNormalization(name='batch_i1')(convi_1)
    convi_1 = Activation('relu')(convi_1)
    convi_2 = Conv2D(128, (3, 3),   padding='same', name='block_i2')(convi_1)
    convi_2 = BatchNormalization(name='batch_i2')(convi_2)
    convi_2 = Activation('relu')(convi_2)

    uppooli=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convi_2)
    convi_3 = Conv2D(64, (3, 3),   padding='same', name='block_i3',kernel_initializer='TruncatedNormal')(convi_2)
    convi_3 = BatchNormalization(name='batch_i3')(convi_3)
    convi_3 = Activation('relu')(convi_3)
    
    #cross 4
    j = concatenate([conva_2,uppooli],axis=-1)
    convj_1 = Conv2D(64, (3, 3),   padding='same',name='block_j1')(j)
    convj_1 = BatchNormalization(name='batch_j1')(convj_1)
    convj_1 = Activation('relu')(convj_1)
    convj_2 = Conv2D(64, (3, 3),   padding='same', name='block_j2')(convj_1)
    convj_2 = BatchNormalization(name='batch_j2')(convj_2)
    convj_2 = Activation('relu')(convj_2)
###############################################################################
    RGBDep=concatenate([inputs,input_depth],axis=-1)
    convk_1 = Conv2D(64, (3, 3), padding='same',name='block_k1')(RGBDep)
    convk_1 = BatchNormalization(name='batch_k1')(convk_1)
    convk_1 = Activation('relu')(convk_1)
    convk_2 = Conv2D(64, (3, 3), padding='same', name='block_k2')(convk_1)
    convk_2 = BatchNormalization(name='batch_k2')(convk_2)
    convk_2 = Activation('relu')(convk_2)
    maxpoolk = MaxPooling2D((2, 2), strides=(2, 2))(convk_2)

    # Block b
    convl_1 = Conv2D(128, (3, 3), padding='same',name='block_l1')(maxpoolk)
    convl_1 = BatchNormalization(name='batch_l1')(convl_1)
    convl_1 = Activation('relu')(convl_1)
    convl_2 = Conv2D(128, (3, 3), padding='same', name='block_l2')(convl_1)
    convl_2 = BatchNormalization(name='batch_l2')(convl_2)
    convl_2 = Activation('relu')(convl_2)
    maxpooll = MaxPooling2D((2, 2), strides=(2, 2))(convl_2)
    convl_3 = Conv2D(64, (3, 3), padding='same', name='block_l3')(convl_2)
    convl_3 = BatchNormalization(name='batch_l3')(convl_3)
    convl_3 = Activation('relu')(convl_3)
    # Block c
    convm_1 = Conv2D(256, (3, 3), padding='same',  name='block_m1')(maxpooll)
    convm_1 = BatchNormalization(name='batch_m1')(convm_1)
    convm_1 = Activation('relu')(convm_1)
    convm_2 = Conv2D(256, (3, 3), padding='same', name='block_m2')(convm_1)
    convm_2 = BatchNormalization(name='batch_m2')(convm_2)
    convm_2 = Activation('relu')(convm_2)
    convm_3 = Conv2D(256, (3, 3), padding='same', name='block_m3')(convm_2)
    convm_3 = BatchNormalization(name='batch_m3')(convm_3)
    convm_3 = Activation('relu')(convm_3)
    maxpoolm = MaxPooling2D((2, 2),strides=(2, 2))(convm_3)
    convm_4 = Conv2D(64, (3, 3), padding='same', name='block_m4')(convm_3)
    convm_4 = BatchNormalization(name='batch_m4')(convm_4)
    convm_4 = Activation('relu')(convm_4)
    
    # Block d
    convn_1 = Conv2D(512, (3, 3), padding='same', name='block_n1')(maxpoolm)
    convn_1 = BatchNormalization(name='batch_n1')(convn_1)
    convn_1 = Activation('relu')(convn_1)
    convn_2 = Conv2D(512, (3, 3), padding='same', name='block_n2')(convn_1)
    convn_2 = BatchNormalization(name='batch_n2')(convn_2)
    convn_2 = Activation('relu')(convn_2)
    convn_3 = Conv2D(512, (3, 3), padding='same', name='block_n3')(convn_2)
    convn_3 = BatchNormalization(name='batch_n3')(convn_3)
    convn_3 = Activation('relu')(convn_3)
    maxpooln = MaxPooling2D((2, 2), strides=(2, 2))(convn_3)
    convn_4 = Conv2D(64, (3, 3), padding='same', name='block_n4')(convn_3)
    convn_4 = BatchNormalization(name='batch_n4')(convn_4)
    convn_4 = Activation('relu')(convn_4)
    # Block 5
    convo_1 = Conv2D(512, (3, 3),   padding='same',name='block_o1')(maxpooln)
    convo_1 = BatchNormalization(name='batch_o1')(convo_1)
    convo_1 = Activation('relu')(convo_1)
    convo_2 = Conv2D(512, (3, 3),   padding='same', name='block_o2')(convo_1)
    convo_2 = BatchNormalization(name='batch_o2')(convo_2)
    convo_2 = Activation('relu')(convo_2)
    convo_3 = Conv2D(512, (3, 3),   padding='same', name='block_o3')(convo_2)
    convo_3 = BatchNormalization(name='batch_o3')(convo_3)
    convo_3 = Activation('relu')(convo_3)

    #de conv begin f
    convp_1 = Conv2D(512, (3, 3),   padding='same', name='block_p1')(convo_3)
    convp_1 = BatchNormalization(name='batch_p1')(convp_1)
    convp_1 = Activation('relu')(convp_1)
    uppoolp=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convp_1)

    #cross 1 g
    q = concatenate([convn_4,uppoolp],axis=-1)
    convq_1 = Conv2D(512, (3, 3),   padding='same', name='block_q1')(q)
    convq_1 = BatchNormalization(name='batch_q1')(convq_1)
    convq_1 = Activation('relu')(convq_1)
    convq_2 = Conv2D(512, (3, 3),   padding='same', name='block_q2')(convq_1)
    convq_2 = BatchNormalization(name='batch_q2')(convq_2)
    convq_2 = Activation('relu')(convq_2)
    convq_3 = Conv2D(64, (3, 3),   padding='same', name='block_q3')(convq_2)
    convq_3 = BatchNormalization(name='batch_q3')(convq_3)
    convq_3 = Activation('relu')(convq_3)
    
    uppoolq=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convq_3)
    #cross 2 h
    r = concatenate([convm_4,uppoolq],axis=-1)
    convr_1 = Conv2D(256, (3, 3),   padding='same',name='block_r1')(r)
    convr_1 = BatchNormalization(name='batch_r1')(convr_1)
    convr_1 = Activation('relu')(convr_1)
    convr_2 = Conv2D(64, (3, 3),   padding='same', name='block_r2')(convr_1)
    convr_2 = BatchNormalization(name='batch_r2')(convr_2)
    convr_2 = Activation('relu')(convr_2)
    
    uppoolr=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convr_2)  
    #cross 3
    s = concatenate([convl_3,uppoolr],axis=-1)
    convs_1 = Conv2D(128, (3, 3),   padding='same', name='block_s1')(s)
    convs_1 = BatchNormalization(name='batch_s1')(convs_1)
    convs_1 = Activation('relu')(convs_1)
    convs_2 = Conv2D(64, (3, 3),   padding='same', name='block_s2')(convs_1)
    convs_2 = BatchNormalization(name='batch_s2')(convs_2)
    convs_2 = Activation('relu')(convs_2)

    uppools=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convs_2)
    #cross 4
    t = concatenate([convk_2,uppools],axis=-1)
    convt_1 = Conv2D(64, (3, 3),   padding='same',name='block_t1')(t)
    convt_1 = BatchNormalization(name='batch_t1')(convt_1)
    convt_1 = Activation('relu')(convt_1)
    convt_2 = Conv2D(64, (3, 3),   padding='same', name='block_t2')(convt_1)
    convt_2 = BatchNormalization(name='batch_t2')(convt_2)
    convt_2 = Activation('relu')(convt_2)
    W=Conv2D(1, (3, 3),padding='same',name='W',activation='sigmoid')(convt_2)
###############################################################################
    fuse1=concatenate([conv5_3,conve_3])
    #fuse conv
    fuse1 = Conv2D(512, (3, 3),   padding='same', name='block_fuse1',kernel_initializer='TruncatedNormal')(fuse1)
    fuse1 = BatchNormalization(name='batch_fuse1')(fuse1)
    fuse1 = Activation('relu')(fuse1)
    
    #de conv begin
    conv6_1 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_1')(fuse1)
    conv6_1 = BatchNormalization()(conv6_1)
    conv6_1 = Activation('relu')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_2')(conv6_1)
    conv6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Activation('relu')(conv6_2)
    conv6_3 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_3')(conv6_2)
    conv6_3 = BatchNormalization()(conv6_3)
    conv6_3 = Activation('relu')(conv6_3)

    uppool1=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv6_3)

    fuse2 = concatenate([conv4_4,convd_4],axis=-1)
    ###########################################################################
    #fuse conv
    fuse2 = Conv2D(512, (3, 3),   padding='same', name='block_fuse2',kernel_initializer='TruncatedNormal')(fuse2)
    fuse2 = BatchNormalization(name='batch_fuse2')(fuse2)
    fuse2 = Activation('relu')(fuse2)
    fuse2=concatenate([fuse2,uppool1])
    ###########################################################################
    conv_m1 = Conv2D(512, (3, 3),   padding='same')(fuse2)
    conv_m1 = BatchNormalization()(conv_m1)
    conv_m1 = Activation('relu')(conv_m1)
    conv7_1 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_1')(conv_m1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = Activation('relu')(conv7_1)
    conv7_2 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_2')(conv7_1)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = Activation('relu')(conv7_2)
    
    uppool2=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv7_2)

    conv7_3 = Conv2D(64, (3, 3),   padding='same', name='block_conv7_3',kernel_initializer='TruncatedNormal')(conv7_2)
    conv7_3 = BatchNormalization(name='batch_conv7_3')(conv7_3)
    conv7_3 = Activation('relu')(conv7_3)

    fuse3 = concatenate([conv3_4,convc_4],axis=-1)
    ###########################################################################
    #fuse conv
    fuse3 = Conv2D(256, (3, 3),   padding='same', name='block_fuse3',kernel_initializer='TruncatedNormal')(fuse3)
    fuse3 = BatchNormalization(name='batch_fuse3')(fuse3)
    fuse3 = Activation('relu')(fuse3)
    fuse3=concatenate([fuse3,uppool2])
    ###########################################################################
    conv_m2 = Conv2D(256, (3, 3),   padding='same', name='de_conv_m2',kernel_initializer='TruncatedNormal')(fuse3)
    conv_m2 = BatchNormalization()(conv_m2)
    conv_m2 = Activation('relu')(conv_m2)
    conv8_1 = Conv2D(256, (3, 3),   padding='same', name='de_conv3_1')(conv_m2)
    conv8_1 = BatchNormalization()(conv8_1)
    conv8_1 = Activation('relu')(conv8_1)
    
    uppool3=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv8_1)  

    conv8_2 = Conv2D(64, (3, 3),   padding='same', name='block_conv8_2',kernel_initializer='TruncatedNormal')(conv8_1)
    conv8_2 = BatchNormalization(name='batch_conv8_2')(conv8_2)
    conv8_2 = Activation('relu')(conv8_2)
    
    fuse4 = concatenate([conv2_2,convb_2],axis=-1)
    ###########################################################################
    #fuse conv
    fuse4 = Conv2D(128, (3, 3),   padding='same', name='block_fuse4',kernel_initializer='TruncatedNormal')(fuse4)
    fuse4 = BatchNormalization(name='batch_fuse4')(fuse4)
    fuse4 = Activation('relu')(fuse4)
    fuse4=concatenate([fuse4,uppool3])
    ###########################################################################
    conv_m3 = Conv2D(128, (3, 3),   padding='same', name='de_conv_m3',kernel_initializer='TruncatedNormal')(fuse4)
    conv_m3 = BatchNormalization()(conv_m3)
    conv_m3 = Activation('relu')(conv_m3)
    conv9_1 = Conv2D(128, (3, 3),   padding='same', name='de_conv2_1')(conv_m3)
    conv9_1 = BatchNormalization()(conv9_1)
    conv9_1 = Activation('relu')(conv9_1)
    
    uppool4=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv9_1)

    conv9_2 = Conv2D(64, (3, 3),   padding='same', name='block_conv9_2',kernel_initializer='TruncatedNormal')(conv9_1)
    conv9_2 = BatchNormalization(name='batch_conv9_2')(conv9_2)
    conv9_2 = Activation('relu')(conv9_2)
    
    fuse5 = concatenate([conv1_2,conva_2],axis=-1)
    ###########################################################################
    #fuse conv
    fuse5 = Conv2D(64, (3, 3),   padding='same', name='fuse_conv5',kernel_initializer='TruncatedNormal')(fuse5)
    fuse5 = BatchNormalization(name='bn_fuse5')(fuse5)
    fuse5 = Activation('relu')(fuse5)
    fuse5=concatenate([fuse5,uppool4])
    ###########################################################################
    conv_m4 = Conv2D(64, (3, 3),   padding='same', name='de_conv_m4',kernel_initializer='TruncatedNormal')(fuse5)
    conv_m4 = BatchNormalization()(conv_m4)
    conv_m4 = Activation('relu')(conv_m4)
    conv10_2 = Conv2D(64, (3, 3),   padding='same', name='de_conv1_2')(conv_m4)
    conv10_2 = BatchNormalization()(conv10_2)
    conv10_2 = Activation('relu')(conv10_2)
###############################################################################
    w28 = tf.image.resize_images(W,[28, 28])
    weight1 = Lambda(lambda x:x*w28)(convg_4)
    weight2 = Lambda(lambda x:x*(1.0-w28))(conv7_3)
    ADD1 = Add()([weight1,weight2])
    
    w56 = tf.image.resize_images(W,[56, 56])
    weight3 = Lambda(lambda x:x*w56)(convh_3)
    weight4 = Lambda(lambda x:x*(1.0-w56))(conv8_2)
    ADD2 = Add()([weight3,weight4])
    
    w112 = tf.image.resize_images(W,[112, 112])
    weight5 = Lambda(lambda x:x*w112)(convi_3)
    weight6 = Lambda(lambda x:x*(1.0-w112))(conv9_2)
    ADD3 = Add()([weight5,weight6])
    
    weight7 = Multiply()([convj_2,W])
    weight8 = Lambda(lambda x:(1.0-x))(W)
    weight8 = Multiply()([conv10_2,weight8])
    ADD4 = Add()([weight7,weight8])
###############################################################################
    upADD1=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(ADD1)
    fuse6=concatenate([upADD1,ADD2])
    ###########################################################################
    conv_fuse6 = Conv2D(128, (3, 3),   padding='same', name='block_fuse6')(fuse6)
    conv_fuse6 = BatchNormalization(name='batch_fuse6')(conv_fuse6)
    conv_fuse6 = Activation('relu')(conv_fuse6)
    output_fuse6=Conv2D(1, (3, 3),padding='same',name='output_fuse6',activation='sigmoid')(conv_fuse6)
    
    #weight9 = Lambda(lambda x:x*output_fuse6)(conv_fuse6)
    weight9 = Multiply()([conv_fuse6,output_fuse6])
    weight9 = Add()([weight9,conv_fuse6])
    upweight9=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(weight9)
    
    fuse7=concatenate([upweight9,ADD3])
    ###########################################################################
    conv_fuse7 = Conv2D(128, (3, 3),   padding='same', name='block_fuse7')(fuse7)
    conv_fuse7 = BatchNormalization(name='batch_fuse7')(conv_fuse7)
    conv_fuse7 = Activation('relu')(conv_fuse7)
    output_fuse7 = Conv2D(1, (3, 3),padding='same',name='output_fuse7',activation='sigmoid')(conv_fuse7)
    
    #weight10 = Lambda(lambda x:x*output_fuse7)(conv_fuse7)
    weight10 = Multiply()([conv_fuse7,output_fuse7])
    weight10 = Add()([weight10,conv_fuse7])
    upweight10=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(weight10)
    
    fuse8=concatenate([upweight10,ADD4])
    ###########################################################################
    conv_fuse8 = Conv2D(128, (3, 3),   padding='same', name='block_fuse8')(fuse8)
    conv_fuse8 = BatchNormalization(name='batch_fuse8')(conv_fuse8)
    conv_fuse8 = Activation('relu')(conv_fuse8)
    output_fuse8=Conv2D(1, (3, 3),padding='same',name='output_fuse8',activation='sigmoid')(conv_fuse8)

    weight11 = Multiply()([conv_fuse8,output_fuse8])
    weight11 = Add()([weight11,conv_fuse8])
    
    conv_11 = Conv2D(128, (3, 3),   padding='same', name='block_11')(weight11)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation('relu')(conv_11)
    
    output=Conv2D(1, (3, 3),padding='same',name='output',activation='sigmoid')(conv_11)
    model = Model(inputs = [inputs,input_depth], outputs = output)

    adam1=keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='binary_crossentropy',optimizer=adam1,metrics=['acc'])
    return model
