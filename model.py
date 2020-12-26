import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import pickle as pk
import itertools


def bce_loss(y_true, y_pred):
    return - y_true * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
           - (1.0 - y_true) * tf.log(tf.clip_by_value((1.0-y_pred), 1e-8, 1.0))

#######################################################################################
def net(img_width,img_height):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width,3)
    print (K.image_data_format())

    inputs=Input(input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), padding='same',name='block1_conv1')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)
    
    maxpool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), padding='same',name='block2_conv1')(maxpool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    maxpool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(maxpool2)
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
    conv4_1 = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(maxpool3)
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
    conv5_1 = Conv2D(512, (3, 3),   padding='same',name='block5_conv1')(maxpool4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3),   padding='same', name='block5_conv2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Activation('relu')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3),   padding='same', name='block5_conv3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_3 = Activation('relu')(conv5_3)


    inputs_deep = Input((img_height, img_width,3))
    conva_1 = Conv2D(64, (3, 3), padding='same',name='blocka_conv_1')(inputs_deep)
    conva_1 = BatchNormalization(name='batch_normalization_a1')(conva_1)
    conva_1 = Activation('relu')(conva_1)
    conva_2 = Conv2D(64, (3, 3), padding='same', name='blocka_conv_2')(conva_1)
    conva_2 = BatchNormalization(name='batch_normalization_a2')(conva_2)
    conva_2 = Activation('relu')(conva_2)
    maxpoola = MaxPooling2D((2, 2), strides=(2, 2))(conva_2)

    # Block b
    convb_1 = Conv2D(128, (3, 3), padding='same',name='blockb_conv1')(maxpoola)
    convb_1 = BatchNormalization(name='batch_normalization_b1')(convb_1)
    convb_1 = Activation('relu')(convb_1)
    convb_2 = Conv2D(128, (3, 3), padding='same', name='blockb_conv2')(convb_1)
    convb_2 = BatchNormalization(name='batch_normalization_b2')(convb_2)
    convb_2 = Activation('relu')(convb_2)
    maxpoolb = MaxPooling2D((2, 2), strides=(2, 2))(convb_2)

    # Block c
    convc_1 = Conv2D(256, (3, 3), padding='same',  name='blockc_conv1')(maxpoolb)
    convc_1 = BatchNormalization(name='batch_normalization_c1')(convc_1)
    convc_1 = Activation('relu')(convc_1)
    convc_2 = Conv2D(256, (3, 3), padding='same', name='blockc_conv2')(convc_1)
    convc_2 = BatchNormalization(name='batch_normalization_c2')(convc_2)
    convc_2 = Activation('relu')(convc_2)
    convc_3 = Conv2D(256, (3, 3), padding='same', name='blockc_conv3')(convc_2)
    convc_3 = BatchNormalization(name='batch_normalization_c3')(convc_3)
    convc_3 = Activation('relu')(convc_3)
    convc_4 = Conv2D(256, (3, 3), padding='same', name='blockc_conv4')(convc_3)
    convc_4 = BatchNormalization(name='batch_normalization_c4')(convc_4)
    convc_4 = Activation('relu')(convc_4)
    maxpoolc = MaxPooling2D((2, 2),strides=(2, 2))(convc_4)

    # Block d
    convd_1 = Conv2D(512, (3, 3), padding='same', name='blockd_conv1')(maxpoolc)
    convd_1 = BatchNormalization(name='batch_normalization_d1')(convd_1)
    convd_1 = Activation('relu')(convd_1)
    convd_2 = Conv2D(512, (3, 3), padding='same', name='blockd_conv2')(convd_1)
    convd_2 = BatchNormalization(name='batch_normalization_d2')(convd_2)
    convd_2 = Activation('relu')(convd_2)
    convd_3 = Conv2D(512, (3, 3), padding='same', name='blockd_conv3')(convd_2)
    convd_3 = BatchNormalization(name='batch_normalization_d3')(convd_3)
    convd_3 = Activation('relu')(convd_3)
    convd_4 = Conv2D(512, (3, 3), padding='same', name='blockd_conv4',)(convd_3)
    convd_4 = BatchNormalization(name='batch_normalization_d4')(convd_4)
    convd_4 = Activation('relu')(convd_4)
    maxpoold = MaxPooling2D((2, 2), strides=(2, 2))(convd_4)

    # Block 5
    conve_1 = Conv2D(512, (3, 3),   padding='same',name='blocke_conv1')(maxpoold)
    conve_1 = BatchNormalization(name='batch_normalization_e1')(conve_1)
    conve_1 = Activation('relu')(conve_1)
    conve_2 = Conv2D(512, (3, 3),   padding='same', name='blocke_conv2')(conve_1)
    conve_2 = BatchNormalization(name='batch_normalization_e2')(conve_2)
    conve_2 = Activation('relu')(conve_2)
    conve_3 = Conv2D(512, (3, 3),   padding='same', name='blocke_conv3')(conve_2)
    conve_3 = BatchNormalization(name='batch_normalization_e3')(conve_3)
    conve_3 = Activation('relu')(conve_3)

    #de conv begin f
    convf_1 = Conv2D(512, (3, 3),   padding='same', name='de_convf_1')(conve_3)
    convf_1 = BatchNormalization(name='batch_normalization_f1')(convf_1)
    convf_1 = Activation('relu')(convf_1)
    convf_2 = Conv2D(512, (3, 3),   padding='same', name='de_convf_2')(convf_1)
    convf_2 = BatchNormalization(name='batch_normalization_f2')(convf_2)
    convf_2 = Activation('relu')(convf_2)
    convf_3 = Conv2D(512, (3, 3),   padding='same', name='de_convf_3')(convf_2)
    convf_3 = BatchNormalization(name='batch_normalization_f3')(convf_3)
    convf_3 = Activation('relu')(convf_3)
    uppoolf=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convf_3)

    #cross 1 g
    g = concatenate([convd_4,uppoolf],axis=-1)
    conv_g = Conv2D(512, (3, 3),   padding='same', name='de_convg_1')(g)
    conv_g = BatchNormalization(name='batch_normalization_g1')(conv_g)
    conv_g = Activation('relu')(conv_g)
    convg_1 = Conv2D(512, (3, 3),   padding='same', name='de_convg_2')(conv_g)
    convg_1 = BatchNormalization(name='batch_normalization_g2')(convg_1)
    convg_1 = Activation('relu')(convg_1)
    convg_2 = Conv2D(512, (3, 3),   padding='same', name='de_convg_3')(convg_1)
    convg_2 = BatchNormalization(name='batch_normalization_g3')(convg_2)
    convg_2 = Activation('relu')(convg_2)
    uppoolg=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convg_2)

    convg_3 = Conv2D(64, (3, 3),   padding='same', name='de_convg_4',kernel_initializer='TruncatedNormal')(convg_2)
    convg_3 = BatchNormalization(name='batch_normalization_g4')(convg_3)
    convg_3 = Activation('relu')(convg_3)
    #cross 2 h
    h = concatenate([convc_4,uppoolg],axis=-1)
    conv_h = Conv2D(256, (3, 3),   padding='same',name='de_convh_1')(h)
    conv_h = BatchNormalization(name='batch_normalization_h1')(conv_h)
    conv_h = Activation('relu')(conv_h)
    convh_1 = Conv2D(256, (3, 3),   padding='same', name='de_convh_2')(conv_h)
    convh_1 = BatchNormalization(name='batch_normalization_h2')(convh_1)
    convh_1 = Activation('relu')(convh_1)
    uppoolh=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convh_1)  
    
    convh_2 = Conv2D(64, (3, 3),   padding='same', name='de_convh_3',kernel_initializer='TruncatedNormal')(convh_1)
    convh_2 = BatchNormalization(name='batch_normalization_h3')(convh_2)
    convh_2 = Activation('relu')(convh_2)
    
    #cross 3
    i = concatenate([convb_2,uppoolh],axis=-1)
    conv_i = Conv2D(128, (3, 3),   padding='same', name='de_convi_1')(i)
    conv_i = BatchNormalization(name='batch_normalization_i1')(conv_i)
    conv_i = Activation('relu')(conv_i)
    convi_1 = Conv2D(128, (3, 3),   padding='same', name='de_convi_2')(conv_i)
    convi_1 = BatchNormalization(name='batch_normalization_i2')(convi_1)
    convi_1 = Activation('relu')(convi_1)
    uppooli=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convi_1)
    
    convi_2 = Conv2D(64, (3, 3),   padding='same', name='de_convi_3',kernel_initializer='TruncatedNormal')(convi_1)
    convi_2 = BatchNormalization(name='batch_normalization_i3')(convi_2)
    convi_2 = Activation('relu')(convi_2)
    
    #cross 4
    j = concatenate([conva_2,uppooli],axis=-1)
    conv_j = Conv2D(64, (3, 3),   padding='same',name='de_convj_1')(j)
    conv_j = BatchNormalization(name='batch_normalization_j1')(conv_j)
    conv_j = Activation('relu')(conv_j)
    convj_2 = Conv2D(64, (3, 3),   padding='same', name='de_convj_2')(conv_j)
    convj_2 = BatchNormalization(name='batch_normalization_j2')(convj_2)
    convj_2 = Activation('relu')(convj_2)
    ###########################################################################
    fuseRGBDepth=concatenate([inputs,inputs_deep],axis=-1)
    # Block a
    convk_1 = Conv2D(64, (3, 3), padding='same',name='blockk_conv_1')(fuseRGBDepth)
    convk_1 = BatchNormalization(name='batch_normalization_k1')(convk_1)
    convk_1 = Activation('relu')(convk_1)
    convk_2 = Conv2D(64, (3, 3), padding='same', name='blockk_conv_2')(convk_1)
    convk_2 = BatchNormalization(name='batch_normalization_k2')(convk_2)
    convk_2 = Activation('relu')(convk_2)
    maxpoolk = MaxPooling2D((2, 2), strides=(2, 2))(convk_2)

    # Block b
    convl_1 = Conv2D(128, (3, 3), padding='same',name='blockl_conv1')(maxpoolk)
    convl_1 = BatchNormalization(name='batch_normalization_l1')(convl_1)
    convl_1 = Activation('relu')(convl_1)
    convl_2 = Conv2D(128, (3, 3), padding='same', name='blockl_conv2')(convl_1)
    convl_2 = BatchNormalization(name='batch_normalization_l2')(convl_2)
    convl_2 = Activation('relu')(convl_2)
    maxpooll = MaxPooling2D((2, 2), strides=(2, 2))(convl_2)

    # Block c
    convm_1 = Conv2D(256, (3, 3), padding='same', name='blockm_conv1')(maxpooll)
    convm_1 = BatchNormalization(name='batch_normalization_m1')(convm_1)
    convm_1 = Activation('relu')(convm_1)
    convm_2 = Conv2D(256, (3, 3), padding='same', name='blockm_conv2')(convm_1)
    convm_2 = BatchNormalization(name='batch_normalization_m2')(convm_2)
    convm_2 = Activation('relu')(convm_2)
    convm_3 = Conv2D(256, (3, 3), padding='same', name='blockm_conv3')(convm_2)
    convm_3 = BatchNormalization(name='batch_normalization_m3')(convm_3)
    convm_3 = Activation('relu')(convm_3)
    convm_4 = Conv2D(256, (3, 3), padding='same', name='blockm_conv4')(convm_3)
    convm_4 = BatchNormalization(name='batch_normalization_m4')(convm_4)
    convm_4 = Activation('relu')(convm_4)
    maxpoolm = MaxPooling2D((2, 2),strides=(2, 2))(convm_4)

    # Block d
    convn_1 = Conv2D(512, (3, 3), padding='same', name='blockn_conv1')(maxpoolm)
    convn_1 = BatchNormalization(name='batch_normalization_n1')(convn_1)
    convn_1 = Activation('relu')(convn_1)
    convn_2 = Conv2D(512, (3, 3), padding='same', name='blockn_conv2')(convn_1)
    convn_2 = BatchNormalization(name='batch_normalization_n2')(convn_2)
    convn_2 = Activation('relu')(convn_2)
    convn_3 = Conv2D(512, (3, 3), padding='same', name='blockn_conv3')(convn_2)
    convn_3 = BatchNormalization(name='batch_normalization_n3')(convn_3)
    convn_3 = Activation('relu')(convn_3)
    convn_4 = Conv2D(512, (3, 3), padding='same', name='blockn_conv4')(convn_3)
    convn_4 = BatchNormalization(name='batch_normalization_n4')(convn_4)
    convn_4 = Activation('relu')(convn_4)
    maxpooln = MaxPooling2D((2, 2), strides=(2, 2))(convn_4)

    # Block 5
    convo_1 = Conv2D(512, (3, 3),   padding='same',name='blocko_conv1')(maxpooln)
    convo_1 = BatchNormalization(name='batch_normalization_o1')(convo_1)
    convo_1 = Activation('relu')(convo_1)
    convo_2 = Conv2D(512, (3, 3),   padding='same', name='blocko_conv2')(convo_1)
    convo_2 = BatchNormalization(name='batch_normalization_o2')(convo_2)
    convo_2 = Activation('relu')(convo_2)
    convo_3 = Conv2D(512, (3, 3),   padding='same', name='blocko_conv3')(convo_2)
    convo_3 = BatchNormalization(name='batch_normalization_o3')(convo_3)
    convo_3 = Activation('relu')(convo_3)

    #de conv begin f
    convp_1 = Conv2D(512, (3, 3),   padding='same', name='de_convp_1')(convo_3)
    convp_1 = BatchNormalization(name='batch_normalization_p1')(convp_1)
    convp_1 = Activation('relu')(convp_1)
    convp_2 = Conv2D(512, (3, 3),   padding='same', name='de_convp_2')(convp_1)
    convp_2 = BatchNormalization(name='batch_normalization_p2')(convp_2)
    convp_2 = Activation('relu')(convp_2)
    convp_3 = Conv2D(512, (3, 3),   padding='same', name='de_convp_3')(convp_2)
    convp_3 = BatchNormalization(name='batch_normalization_p3')(convp_3)
    convp_3 = Activation('relu')(convp_3)
    uppoolp=UpSampling2D(size=(2, 2),data_format=K.image_data_format())(convp_3)

    #cross 1 g
    q = concatenate([convn_4,uppoolp],axis=-1)
    conv_q = Conv2D(512, (3, 3),   padding='same', name='de_convq_1')(q)
    conv_q = BatchNormalization(name='batch_normalization_q1')(conv_q)
    conv_q = Activation('relu')(conv_q)
    convq_1 = Conv2D(512, (3, 3),   padding='same', name='de_convq_2')(conv_q)
    convq_1 = BatchNormalization(name='batch_normalization_q2')(convq_1)
    convq_1 = Activation('relu')(convq_1)
    convq_2 = Conv2D(512, (3, 3),   padding='same', name='de_convq_3')(convq_1)
    convq_2 = BatchNormalization(name='batch_normalization_q3')(convq_2)
    convq_2 = Activation('relu')(convq_2)
    
    uppoolq=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convq_2)
    uppoolq_1=UpSampling2D(size=(4, 4), data_format=K.image_data_format())(uppoolq)
    outq=Conv2D(1, (3, 3),padding='same',name='outq')(uppoolq_1)

    #cross 2 h
    r = concatenate([convm_4,uppoolq],axis=-1)
    conv_r = Conv2D(256, (3, 3),   padding='same',name='de_convr_1')(r)
    conv_r = BatchNormalization(name='batch_normalization_r1')(conv_r)
    conv_r = Activation('relu')(conv_r)
    convr_1 = Conv2D(256, (3, 3),   padding='same', name='de_convr_2')(conv_r)
    convr_1 = BatchNormalization(name='batch_normalization_r2')(convr_1)
    convr_1 = Activation('relu')(convr_1)
    
    uppoolr=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convr_1)  
    uppoolr_1=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(uppoolr)
    outr=Conv2D(1, (3, 3),padding='same',name='outr')(uppoolr_1)
    
    #cross 3
    s = concatenate([convl_2,uppoolr],axis=-1)
    conv_s = Conv2D(128, (3, 3),   padding='same', name='de_convs_1')(s)
    conv_s = BatchNormalization(name='batch_normalization_s1')(conv_s)
    conv_s = Activation('relu')(conv_s)
    convs_1 = Conv2D(128, (3, 3),   padding='same', name='de_convs_2')(conv_s)
    convs_1 = BatchNormalization(name='batch_normalization_s2')(convs_1)
    convs_1 = Activation('relu')(convs_1)
    
    uppools=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(convs_1)
    outs=Conv2D(1, (3, 3),padding='same',name='outs')(uppools)
    
    #cross 4
    t = concatenate([convk_2,uppools],axis=-1)
    conv_t = Conv2D(64, (3, 3),   padding='same',name='de_convt_1')(t)
    conv_t = BatchNormalization(name='batch_normalization_t1')(conv_t)
    conv_t = Activation('relu')(conv_t)
    convt_2 = Conv2D(64, (3, 3),   padding='same', name='de_convt_2')(conv_t)
    convt_2 = BatchNormalization(name='batch_normalization_t2')(convt_2)
    convt_2 = Activation('relu')(convt_2)
    outt=Conv2D(1, (3, 3),padding='same',name='outt')(convt_2)
    
    con_W=concatenate([outq,outr,outs,outt])

    W=Conv2D(1, (3, 3),padding='same',name='W',activation='sigmoid')(con_W)
    ####################AA#####################################################
    fuse1=concatenate([conv5_3,conve_3])
    #fuse conv
    fuse1 = Conv2D(512, (3, 3),   padding='same', name='fuse_conv1')(fuse1)
    fuse1 = BatchNormalization(name='bn_fuse1')(fuse1)
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

    merge1 = concatenate([conv4_4,convd_4],axis=-1)
    ###########################################################################
    #fuse conv
    fuse2 = Conv2D(512, (3, 3),   padding='same', name='fuse_conv2')(merge1)
    fuse2 = BatchNormalization(name='bn_fuse2')(fuse2)
    fuse2 = Activation('relu')(fuse2)
    finalfuse2=concatenate([fuse2,uppool1])
    ###########################################################################
    conv_m1 = Conv2D(512, (3, 3),   padding='same',name='conv2d_1')(finalfuse2)
    conv_m1 = BatchNormalization()(conv_m1)
    conv_m1 = Activation('relu')(conv_m1)
    conv7_1 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_1')(conv_m1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = Activation('relu')(conv7_1)
    conv7_2 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_2')(conv7_1)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = Activation('relu')(conv7_2)
    
    uppool2=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv7_2)

    conv7_3 = Conv2D(64, (3, 3),   padding='same', name='de_convconv7_3')(conv7_2)
    conv7_3 = BatchNormalization(name='batch_normalization_conv7_3')(conv7_3)
    conv7_3 = Activation('relu')(conv7_3)

    merge2 = concatenate([conv3_4,convc_4],axis=-1)
    ###########################################################################
    #fuse conv
    fuse3 = Conv2D(256, (3, 3),   padding='same', name='fuse_conv3')(merge2)
    fuse3 = BatchNormalization(name='bn_fuse3')(fuse3)
    fuse3 = Activation('relu')(fuse3)
    finalfuse3=concatenate([fuse3,uppool2])
    ###########################################################################
    conv_m2 = Conv2D(256, (3, 3),   padding='same', name='conv2d_3')(finalfuse3)
    conv_m2 = BatchNormalization()(conv_m2)
    conv_m2 = Activation('relu')(conv_m2)
    conv8_1 = Conv2D(256, (3, 3),   padding='same', name='de_conv3_1')(conv_m2)
    conv8_1 = BatchNormalization()(conv8_1)
    conv8_1 = Activation('relu')(conv8_1)
    
    uppool3=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv8_1)  

    conv8_3 = Conv2D(64, (3, 3),   padding='same', name='de_convconv8_3')(conv8_1)
    conv8_3 = BatchNormalization(name='batch_normalization_conv8_3')(conv8_3)
    conv8_3 = Activation('relu')(conv8_3)
    
    merge3 = concatenate([conv2_2,convb_2],axis=-1)
    ###########################################################################
    #fuse conv
    fuse4 = Conv2D(128, (3, 3),   padding='same', name='fuse_conv4')(merge3)
    fuse4 = BatchNormalization(name='bn_fuse4')(fuse4)
    fuse4 = Activation('relu')(fuse4)
    finalfuse4=concatenate([fuse4,uppool3])
    ###########################################################################
    conv_m3 = Conv2D(128, (3, 3),   padding='same', name='conv2d_5')(finalfuse4)
    conv_m3 = BatchNormalization()(conv_m3)
    conv_m3 = Activation('relu')(conv_m3)
    conv9_1 = Conv2D(128, (3, 3),   padding='same', name='de_conv2_1')(conv_m3)
    conv9_1 = BatchNormalization()(conv9_1)
    conv9_1 = Activation('relu')(conv9_1)
    
    uppool4=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv9_1)

    conv9_3 = Conv2D(64, (3, 3),   padding='same', name='de_convconv9_3')(conv9_1)
    conv9_3 = BatchNormalization(name='batch_normalization_conv9_3')(conv9_3)
    conv9_3 = Activation('relu')(conv9_3)
    
    merge4 = concatenate([conv1_2,conva_2],axis=-1)
    ###########################################################################
    #fuse conv
    fuse5 = Conv2D(64, (3, 3),   padding='same', name='fuse_conv5')(merge4)
    fuse5 = BatchNormalization(name='bn_fuse5')(fuse5)
    fuse5 = Activation('relu')(fuse5)
    finalfuse5=concatenate([fuse5,uppool4])
    ###########################################################################
    conv_m4 = Conv2D(64, (3, 3),   padding='same', name='conv2d_7')(finalfuse5)
    conv_m4 = BatchNormalization()(conv_m4)
    conv_m4 = Activation('relu')(conv_m4)
    conv10_2 = Conv2D(64, (3, 3),   padding='same', name='de_conv1_2')(conv_m4)
    conv10_2 = BatchNormalization()(conv10_2)
    conv10_2 = Activation('relu')(conv10_2)
    #outputs=Conv2D(1, (3, 3),padding='same',name='conv2d_8')(conv10_2)
    conv10_3 = Conv2D(64, (3, 3),   padding='same',dilation_rate=(3, 3),  name='de_conv10_3')(conv10_2)
    conv10_3 = BatchNormalization(name='batch_normalization_conv10_3')(conv10_3)
    conv10_3 = Activation('relu')(conv10_3)
    maxpoolconv10_3 = MaxPooling2D((2, 2), strides=(2, 2), name='maxpoolconv10_3')(conv10_3)###112*112
    
    conv10_4 = Conv2D(64, (3, 3),   padding='same',dilation_rate=(5, 5),  name='de_conv10_4')(maxpoolconv10_3)
    conv10_4 = BatchNormalization(name='batch_normalization_conv10_4')(conv10_4)
    conv10_4 = Activation('relu')(conv10_4)
    maxpoolconv10_4 = MaxPooling2D((2, 2), strides=(2, 2), name='maxpoolconv10_4')(conv10_4)###56*56
    
    conv10_5 = Conv2D(64, (3, 3),   padding='same',dilation_rate=(7, 7),  name='de_conv10_5')(maxpoolconv10_4)
    conv10_5 = BatchNormalization(name='batch_normalization_conv10_5')(conv10_5)
    conv10_5 = Activation('relu')(conv10_5)###56*56
    ###########################################################################
    ###########################################################################
    w28 = tf.image.resize_images(W,[28, 28])
    weight_1 = Lambda(lambda x:x*w28)
    weight_2 = Lambda(lambda x:x*(1.0-w28))
    weight_gru1 = weight_1(convg_3)
    weight_gru2 = weight_2(conv7_3)
    ADD1 = Add()([weight_gru1,weight_gru2])###28
    
    w56 = tf.image.resize_images(W,[56, 56])
    weight_3 = Lambda(lambda x:x*w56)
    weight_4 = Lambda(lambda x:x*(1.0-w56))
    weight_gru3 = weight_3(convh_2)
    weight_gru4 = weight_4(conv8_3)
    ADD2 = Add()([weight_gru3,weight_gru4])####56
    
    w112 = tf.image.resize_images(W,[112, 112])
    weight_5 = Lambda(lambda x:x*w112)
    weight_6 = Lambda(lambda x:x*(1.0-w112))
    weight_gru5 = weight_5(convi_2)
    weight_gru6 = weight_6(conv9_3)
    ADD3 = Add()([weight_gru5,weight_gru6])###112
    
    weight_8 = Lambda(lambda x:x*(1.0-W))
    weight_gru8 = weight_8(conv10_2)
    weight_gru7 = Multiply()([convj_2,W])
    ADD4 = Add()([weight_gru7,weight_gru8])###224
    
    
    uppoolADD1=UpSampling2D(size=(2, 2),name='uppoolADD1', data_format=K.image_data_format())(ADD1)
    #uppoolADD1_1=UpSampling2D(size=(4, 4),name='uppoolADD1_1', data_format=K.image_data_format())(uppoolADD1)
    
    ADD56=concatenate([uppoolADD1,ADD2,conv10_5])
    ###########################################################################
    conv_ADD56 = Conv2D(192, (3, 3),   padding='same', name='convconv_ADD56')(ADD56)
    conv_ADD56 = BatchNormalization(name='batch_normalization_conv_ADD56')(conv_ADD56)
    conv_ADD56 = Activation('relu')(conv_ADD56)
    outputs_conv_ADD56=Conv2D(1, (3, 3),padding='same',name='outputs_conv_ADD56',activation='sigmoid')(conv_ADD56)
    
    weight_conv_ADD56 = Lambda(lambda x:x*outputs_conv_ADD56)
    weight_gruconv_ADD56 = weight_conv_ADD56(conv_ADD56)
    ADDADD56 = Add()([weight_gruconv_ADD56,conv_ADD56])
    uppoolADD2=UpSampling2D(size=(2, 2),name='uppoolADD2', data_format=K.image_data_format())(ADDADD56)
    
    ADD567=concatenate([uppoolADD2,ADD3,conv10_4])
    ###########################################################################
    conv_ADD567 = Conv2D(192, (3, 3),   padding='same', name='convconv_ADD567')(ADD567)
    conv_ADD567 = BatchNormalization(name='batch_normalization_conv_ADD567')(conv_ADD567)
    conv_ADD567 = Activation('relu')(conv_ADD567)
    outputs_conv_ADD567=Conv2D(1, (3, 3),padding='same',name='outputs_conv_ADD567',activation='sigmoid')(conv_ADD567)
    
    weight_conv_ADD567 = Lambda(lambda x:x*outputs_conv_ADD567)
    weight_gruconv_ADD567 = weight_conv_ADD567(conv_ADD567)
    ADDADD567 = Add()([weight_gruconv_ADD567,conv_ADD567])
    uppoolADD3=UpSampling2D(size=(2, 2),name='uppoolADD3', data_format=K.image_data_format())(ADDADD567)
    
    ADD5678=concatenate([uppoolADD3,ADD4,conv10_3])
    ###########################################################################
    conv_ADD5678 = Conv2D(192, (3, 3),   padding='same', name='convconv_ADD5678')(ADD5678)
    conv_ADD5678 = BatchNormalization(name='batch_normalization_conv_ADD5678')(conv_ADD5678)
    conv_ADD5678 = Activation('relu')(conv_ADD5678)
    outputs_conv_ADD5678=Conv2D(1, (3, 3),padding='same',name='outputs_conv_ADD5678',activation='sigmoid')(conv_ADD5678)
    
    weight_conv_ADD5678 = Lambda(lambda x:x*outputs_conv_ADD5678)
    weight_gruconv_ADD5678 = weight_conv_ADD5678(conv_ADD5678)
    ADDADD5678 = Add()([weight_gruconv_ADD5678,conv_ADD5678])
    
    conv_ADDADD5678 = Conv2D(192, (3, 3),   padding='same', name='convconv_conv_ADDADD5678')(ADDADD5678)
    conv_ADDADD5678 = BatchNormalization(name='batch_normalization_conv_conv_ADDADD5678')(conv_ADDADD5678)
    conv_ADDADD5678 = Activation('relu')(conv_ADDADD5678)
    
    outputs_all=Conv2D(1, (3, 3),padding='same',name='outputs_all',activation='sigmoid')(conv_ADDADD5678)
    model = Model(inputs = [inputs,inputs_deep], outputs = outputs_all)

    adam1=keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss= 'binary_crossentropy',optimizer=adam1,metrics=['acc'])
    return model