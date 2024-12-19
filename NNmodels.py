import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import matplotlib.pyplot as plt

import sys
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.layers import Conv2D

from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models, regularizers


def residual_block(x, filters, kernel_size=(3, 3)):
    """Defines a residual block with two convolutional layers."""
    shortcut = x
    # Project shortcut to the same shape if needed
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same',
                                 kernel_regularizer=regularizers.l2(0.01))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Convolutional path
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut and convolutions
    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def CNN(input_shape, num_classes=2):
    """Creates and returns a residual CNN model."""
    
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


def BasicModel(input_shape, num_classes=2):
    """
    Function to create a convolutional neural network model.
    
    Args:
    - input_shape: tuple, shape of the input image (height, width, channels)
    - num_classes: int, number of output classes (for classification, e.g., road vs. non-road)
    
    Returns:
    - model: A compiled Keras model.
    """
    
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for one-hot encoding
    
    # Model summary
    model.summary()
    
    return model