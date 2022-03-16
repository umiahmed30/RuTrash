import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, AveragePooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
import sys
import scipy
from scipy import signal
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import warnings



DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
BATCH_SIZE = 16

if __name__ == "__main__":

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(rotation_range = 45, horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    input_size = 224
    input_channels= 3



    classes = list(iter(batches.class_indices))
    base_model = ResNet50(include_top = False,weights="imagenet",input_shape=(input_size,input_size,input_channels))
    x = AveragePooling2D((7,7),name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(len(classes),activation="softmax")(x)
    finetuned_model = Model(base_model.input,x)
    for layer in base_model.layers:
    	layer.trainable = False
    finetuned_model.save_weights("all_nontrainable50.h5")

    base_model = ResNet50(include_top = False,weights="imagenet",input_shape=(input_size,input_size,input_channels))
    x = AveragePooling2D((7,7),name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(len(classes),activation="softmax")(x)


    finetuned_model = Model(base_model.input,x)

    for layer in base_model.layers[:10]:
    	layer.trainable = False
    finetuned_model.load_weights("all_nontrainable50.h5")

    finetuned_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint('resnet50_best_train_epoch_30.h5', verbose=1, save_best_only=True)

    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=30, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.save('resnet50_train_epoch_30.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(acc)
    print(loss)
    print(val_acc)
    print(val_loss)
    ep = range(len(acc))
    plt.plot(ep, val_loss,'b',label = 'Valid loss')
    plt.plot(ep,loss,'r',label='Training loss')
    plt.plot('T and V Loss')
    plt.legend()
    plt.show()
