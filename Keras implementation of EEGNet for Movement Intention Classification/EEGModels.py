# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:40:49 2020

@author: Karan
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.models import Model


def EEGModel(input_shape,DrRate,N_channels,N_temp_filters,N_spatial_filters,temp_filter_len,av_pool_size,av_pool_stride,final_conv_filters):
    
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # TEMPORAL CONV
    X = Conv2D(N_temp_filters, (temp_filter_len, 1), strides = (1, 1), name = 'conv0',padding = 'valid')(X_input)
    
    # SPATIAL CONV
    X = Conv2D(N_spatial_filters,  (1,N_channels), padding = 'valid',name = 'conv1',use_bias= False)(X)
   
    # BATCH NORM (If axis is not defined a spatial batch normalisation is done, normalizing each channel and not normalizing each batch)
    X = BatchNormalization(momentum = 0.1)(X)
    
    # SQUARING
    X = tf.keras.backend.square(X)
    
    # AVPOOL
    X = AveragePooling2D(pool_size=(av_pool_size, 1), strides = (av_pool_stride,1), padding="valid", name = "pool0")(X)
    
    # LOG ACTIVATION
    X = tf.keras.backend.log(X) 
    
    # DROPOUT 
    X = Dropout(DrRate)(X)
    
    # FINDING SIZE OF TENSOR FOR RESHAPING
    Y = tf.keras.backend.int_shape(X)
    
    # TENSOR RESHAPE
    X = tf.keras.layers.Reshape((Y[1],N_spatial_filters,1), input_shape=(Y[1],1,N_spatial_filters))(X)
    
    # FINAL CONV
    X = Conv2D(final_conv_filters,  (Y[1],1), padding = 'valid',name = 'conv2')(X)
    
    # SOFTMAX CLASSIFICATION
    X = Flatten()(X)
    X = Dense(4 ,activation = "softmax")(X)
    

    model = Model(inputs = X_input, outputs = X)
    
    return model


