
from helpers import *
from constants import *
from NNmodels import *
import tensorflow as tf
from tensorflow.keras import  callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

data_dir = os.getcwd() + '/dataset/training/'
data_filename = data_dir + "images/"
labels_filename = data_dir + "groundtruth/"

data, labels = load_data(data_filename, labels_filename, TRAINING_SIZE)


model = CNN(input_shape=(100, 100, 3), num_classes=2)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = callbacks.EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

lr_callback = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5,
                                verbose=1, mode='auto', min_delta=0, cooldown=0, min_lr=0)


train_generator = data_generators(data, labels, 100, batch_size=32)


history = model.fit(train_generator,steps_per_epoch=int((len(data))),
                    epochs=100,
                    callbacks=[early_stopping, lr_callback],
                    verbose=1)


