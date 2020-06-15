import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks





# Shallow Convnet Model:

def EEGModel(input_shape,DrRate,N_channels,N_temp_filters,N_spatial_filters,temp_filter_len,av_pool_size,av_pool_stride,final_conv_filters):
    
    
    # INPUT LAYER
    X_input = Input(input_shape)

    # TEMPORAL CONV
    X = Conv2D(N_temp_filters, (temp_filter_len, 1), strides = (1, 1), name = 'Temporal_convolution',padding = 'valid')(X_input)
    
    # SPATIAL CONV
    X = Conv2D(N_spatial_filters,  (1,N_channels), padding = 'valid',name = 'Spatial_convolution',use_bias= False)(X)
   
    # BATCH NORM 
    X = BatchNormalization(momentum = 0.1)(X)
    
    # SQUARING
    X = tf.keras.backend.square(X)
    
    # AVPOOL
    X = AveragePooling2D(pool_size=(av_pool_size, 1), strides = (av_pool_stride,1), padding="valid", name = "Average_pooling")(X)
    
    # LOG ACTIVATION
    X = tf.keras.backend.log(X) 
    
    # DROPOUT 
    X = Dropout(DrRate,name="Dropout_layer")(X)
    
    # FINDING SIZE OF TENSOR FOR RESHAPING
    Y = tf.keras.backend.int_shape(X)
    
    # TENSOR RESHAPE
    X = tf.keras.layers.Reshape((Y[1],N_spatial_filters,1), input_shape=(Y[1],1,N_spatial_filters))(X)
    
    # FINAL CONV
    X = Conv2D(final_conv_filters,  (Y[1],1), padding = 'valid',name = 'Final_convolution')(X)
    
    # SOFTMAX CLASSIFICATIONS
    X = Flatten()(X)
    X = Dense(4 ,activation = "softmax",name = "Output_layer")(X)
    

    model = Model(inputs = X_input, outputs = X)
    
    return model



# Callback 1: Stop training when validation accuracy doesnt increase for X amount 
# of epochs

def Callback_Iteration1(patience):
    Val_acc_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max',
                                           baseline=None, restore_best_weights=True)
    
    return Val_acc_stop



            



