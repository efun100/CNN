import os
import cv2
import random
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.mobilenetv2 import MobileNetV2
# from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD

# from keras.applications.mobilenet_v2 import MobileNetV2

dirs = os.listdir("data/train")
dir_count = len(dirs)
BATCH_SIZE = 16


def Image_Classification_model(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50, RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1
    base_model = MobileNetV2(include_top=False, input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    x = base_model.output
    # let's add a fully-connected layer
    x = Flatten()(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


model = Image_Classification_model(nb_classes=dir_count, img_rows=224, img_cols=224)

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9))

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=BATCH_SIZE)

filepath = "weights-improvement-best.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
callbacks_list = [ReduceLROnPlateau(patience=5,verbose=1), checkpoint, EarlyStopping(patience=50, verbose=1)]

model.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=500,
    validation_data=validation_generator,
    validation_steps=80,
    callbacks=callbacks_list)

model.save('./my_transfer.h5')
