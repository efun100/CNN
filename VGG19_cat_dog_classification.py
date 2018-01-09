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
import random

from keras.optimizers import SGD
from keras.utils import np_utils

import os

ROW = 100
COL = 100


# There are 40 different classes
# input image dimensions  
# number of convolutional filters to use  

def readClass(dirPath):
    classDirPathList = []
    dirName = os.listdir(dirPath)
    for d in dirName:
        classDirPath = os.path.join(dirPath, d)
        classDirPathList.append(classDirPath)
    # print(classDirPathList)
    return classDirPathList


def load_data_RGB(dirPath1, img_rows, img_cols):
    nb_classes = 0
    classDirPathList = readClass(dirPath1)
    numpy.save("class.npy", classDirPathList)
    print("class.npy saved")
    trainNum = 0
    testNum = 0
    valNum = 0
    trainLabelList = []
    trainImageList = []
    testLabelList = []
    testImageList = []
    valLabelList = []
    valImageList = []
    # print(classDirPathList)
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 3)):
                ran = random.randint(1, 22);
                if ran == 1:
                    testLabelList.append(nb_classes)
                    testImageList.append(img_ndarray)
                    testNum = testNum + 1
                elif ran == 2:
                    valLabelList.append(nb_classes)
                    valImageList.append(img_ndarray)
                    valNum = valNum + 1
                else:
                    trainLabelList.append(nb_classes)
                    trainImageList.append(img_ndarray)
                    trainNum = trainNum + 1

        nb_classes = nb_classes + 1
    trainFeatureNumpy = numpy.array(trainImageList)
    trainLabelNumpy = numpy.array(trainLabelList)
    testFeatureNumpy = numpy.array(testImageList)
    testLabelNumpy = numpy.array(testLabelList)
    valFeatureNumpy = numpy.array(valImageList)
    valLabelNumpy = numpy.array(valLabelList)
    print("testNum = %d" % testNum)
    print("valNum = %d" % valNum)
    print("trainNum = %d" % trainNum)

    trainLabelNumpy = trainLabelNumpy.astype('int64')
    testLabelNumpy = testLabelNumpy.astype('int64')
    valLabelNumpy = valLabelNumpy.astype('int64')
    trainLabelNumpy = np_utils.to_categorical(trainLabelNumpy, nb_classes)
    valLabelNumpy = np_utils.to_categorical(valLabelNumpy, nb_classes)

    trainFeatureNumpy = trainFeatureNumpy.reshape(trainFeatureNumpy.shape[0], img_rows, img_cols, 3)
    testFeatureNumpy = testFeatureNumpy.reshape(testFeatureNumpy.shape[0], img_rows, img_cols, 3)
    valFeatureNumpy = valFeatureNumpy.reshape(valFeatureNumpy.shape[0], img_rows, img_cols, 3)

    return (trainFeatureNumpy, trainLabelNumpy), (testFeatureNumpy, testLabelNumpy), (
        valFeatureNumpy, valLabelNumpy), nb_classes


def load_predict_data_RGB(dataPath, img_rows, img_cols):
    imageList = []
    img = Image.open(dataPath)
    img = img.resize((img_rows, img_cols))
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    if (img_ndarray.shape == (img_rows, img_cols, 3)):
        imageList.append(img_ndarray)
    imageList = numpy.array(imageList)
    imageList = imageList.reshape(imageList.shape[0], img_rows, img_cols, 3)
    return imageList


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
    pic = load_predict_data_RGB(model_url, ROW, COL)
    result = predict_model(pic, 'model_weights5.h5')
    print(model_url)
    print(result)
    print(numpy.argmax(result))
    print(a[numpy.argmax(result)])
    print("\n")
    print("")


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), (X_val, y_val), class_num = load_data_RGB('./lfw', ROW, COL)

    # print("class_num = %d" % class_num)

    #a = numpy.load("class.npy")
    #print(a)
    #class_num = a.size
    #print(class_num)

    model = Image_Classification_model(nb_classes=class_num, img_rows=ROW, img_cols=COL)
    model.load_weights('model_weights5.h5')

    # train_model(model, X_train, y_train, X_val, y_val, 64, 10, 'model_weights1.h5')
    # test_model(X_test, y_test, 'model_weights1.h5')
    # train_model(model, X_train, y_train, X_val, y_val, 64, 10, 'model_weights2.h5')
    # test_model(X_test, y_test, 'model_weights2.h5')
    # train_model(model, X_train, y_train, X_val, y_val, 64, 10, 'model_weights3.h5')
    # test_model(X_test, y_test, 'model_weights3.h5')
    # train_model(model, X_train, y_train, X_val, y_val, 64, 10, 'model_weights4.h5')
    # test_model(X_test, y_test, 'model_weights4.h5')
    # train_model(model, X_train, y_train, X_val, y_val, 64, 10, 'model_weights5.h5')
    test_model(X_test, y_test, 'model_weights5.h5')
    quit()
    #model.load_weights('model_weights5.h5')

'''
    print_result(a, './lfw/Abdullah_al-Attiyah/Abdullah_al-Attiyah_0001.jpg')
    print_result(a, './lfw/Beth_Blough/Beth_Blough_0001.jpg')
    print_result(a, './lfw/Chang_Sang/Chang_Sang_0001.jpg')


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
'''
