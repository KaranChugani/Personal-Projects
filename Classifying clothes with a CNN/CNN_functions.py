# -*- coding: utf-8 -*-
"""
This function returns a CNN model and takes as an input the optimizer function 
to be used
"""
import tensorflow as tf

def create_model(function):
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=function, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
