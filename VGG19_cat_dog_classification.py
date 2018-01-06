'''''Train a simple convnet on the part olivetti faces dataset. 
 
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py 
 
Get to 95% test accuracy after 25 epochs (there is still a lot of margin for parameter tuning). 
'''

# from __future__ import print_function
import numpy

numpy.random.seed(1337)  # for reproducibility  

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras import backend as K

from keras.optimizers import SGD
from keras.utils import np_utils

import os


# There are 40 different classes  
# input image dimensions  
# number of convolutional filters to use  

def readClass(dirPath):
    classDirPathList = []
    dirName = os.listdir(dirPath)
    for d in dirName:
        classDirPath = os.path.join(dirPath, d)
        classDirPathList.append(classDirPath)
    return classDirPathList


def load_data_RGB(dirPath1, img_rows, img_cols, nb_classes):
    count = 0
    classDirPathList = readClass(dirPath1)
    numpy.save("class.npy", classDirPathList)
    print("class.npy saved")
    labelList = []
    imageList = []
    print(classDirPathList)
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 3)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy1 = numpy.array(imageList)
    labelNumpy1 = numpy.array(labelList)
    print(labelNumpy1)

    labelNumpy1 = labelNumpy1.astype('int64')
    if nb_classes != 1:
        labelNumpy1 = np_utils.to_categorical(labelNumpy1, nb_classes)

    featureNumpy1 = featureNumpy1.reshape(featureNumpy1.shape[0], img_rows, img_cols, 3)

    rval = (featureNumpy1, labelNumpy1)
    print(labelNumpy1)
    return rval


def load_predict_data_RGB(dataPath, img_rows, img_cols, nb_classes):
    imageList = []
    img = Image.open(dataPath)
    img = img.resize((img_rows, img_cols))
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    if (img_ndarray.shape == (img_rows, img_cols, 3)):
        imageList.append(img_ndarray)
    imageList = numpy.array(imageList)
    imageList = imageList.reshape(imageList.shape[0], img_rows, img_cols, 3)
    return imageList


def load_data_Grey(dirPath1, dirPath2, dirPath3, img_rows, img_cols, nb_classes):
    count = 0
    classDirPathList = readClass(dirPath1)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy1 = numpy.array(imageList)
    labelNumpy1 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath2)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy2 = numpy.array(imageList)
    labelNumpy2 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath3)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy3 = numpy.array(imageList)
    labelNumpy3 = numpy.array(labelList)

    labelNumpy1 = labelNumpy1.astype('int64')
    labelNumpy2 = labelNumpy2.astype('int64')
    labelNumpy3 = labelNumpy3.astype('int64')

    labelNumpy1 = np_utils.to_categorical(labelNumpy1, nb_classes)
    labelNumpy2 = np_utils.to_categorical(labelNumpy2, nb_classes)
    # labelNumpy3 = np_utils.to_categorical(labelNumpy3, nb_classes)

    featureNumpy1 = featureNumpy1.reshape(featureNumpy1.shape[0], img_rows, img_cols, 3)
    featureNumpy2 = featureNumpy2.reshape(featureNumpy2.shape[0], img_rows, img_cols, 3)
    featureNumpy3 = featureNumpy3.reshape(featureNumpy3.shape[0], img_rows, img_cols, 3)

    rval = [(featureNumpy1, labelNumpy1), (featureNumpy2, labelNumpy2), (featureNumpy3, labelNumpy3)]

    return rval


def Image_Classification_model(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50, RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, color),
                       classes=nb_classes)

    x = base_model.output
    # let's add a fully-connected layer
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, model_url):
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_val, Y_val),
              shuffle=True)
    model.save_weights(model_url, overwrite=True)
    return model


def test_model(X, Y, model_url):
    model.load_weights(model_url)
    classes = numpy.argmax(model.predict(X), axis=1)
    test_accuracy = numpy.mean(numpy.equal(Y, classes))
    print("test accuarcy:", test_accuracy)


def predict_model(X, model_url):
    model.load_weights(model_url)
    classes = model.predict(X, verbose=0)
    return classes


def print_result(a, model_url):
    pic = load_predict_data_RGB(model_url, 50, 50, 2)
    result = predict_model(pic, 'model_weights.h5')
    print(model_url)
    print(result)
    print(numpy.argmax(result))
    print(a[numpy.argmax(result)])
    print("\n")
    print("")


if __name__ == '__main__':
    a = numpy.load("class.npy")
    print(a)
    # (X_train, y_train) = load_data_RGB('/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/train', 50, 50, 2)
    # (X_val, y_val) = load_data_RGB('/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/test', 50, 50, 2)
    # (X_test, y_test) = load_data_RGB('/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/test', 50, 50, 1)

    model = Image_Classification_model(nb_classes=2, img_rows=50, img_cols=50)
    model.load_weights('model_weights.h5')

    # train_model(model, X_train, y_train, X_val, y_val, 128, 5, 'model_weights.h5')
    # test_model(X_test, y_test, 'model_weights.h5')

    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/2.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/3.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/4.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/5.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/6.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/7.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/8.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/9.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/11.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/12.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/13.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/14.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/15.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/16.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/17.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/18.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/19.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/20.jpg')
    print_result(a, '/home/hydrogen/PycharmProjects/vgg19_dog_cat/cat_dog_Dataset/submit/21.jpg')
