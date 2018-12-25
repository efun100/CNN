import os
import cv2
import numpy
from keras.models import load_model

model = load_model("my_transfer.h5")

dir_path = "data/train/fist/"
write_path = "imwrite/fist/"
img_rows = 224
img_cols = 224


imageAllFile = os.listdir(dir_path)
for imgPath in imageAllFile[:20]:
    imageList = []
    imagePartList = []
    print(imgPath)
    img = cv2.imread(dir_path + imgPath)
    part_right = int(img.shape[1] * 3 / 5)
    part_down = int(img.shape[0] * 3 / 4)
    img_part = img[:part_down, :part_right, :]
    imagePartList.append(img_part)
    # cv2.imshow("test", img_part)
    # print(img_part.shape)
    # cv2.imwrite("1.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    part_left = int(img.shape[1] * 1 / 5)
    part_right = int(img.shape[1] * 4 / 5)
    part_down = int(img.shape[0] * 3 / 4)
    img_part = img[:part_down, part_left:part_right, :]
    imagePartList.append(img_part)
    # cv2.imshow("test", img_part)
    # print(img_part.shape)
    # cv2.imwrite("2.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    part_left = int(img.shape[1] * 2 / 5)
    part_down = int(img.shape[0] * 3 / 4)
    img_part = img[:part_down, part_left:, :]
    imagePartList.append(img_part)
    # cv2.imshow("test", img_part)
    # print(img_part.shape)
    # cv2.imwrite("3.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    part_top = int(img.shape[0] * 1 / 4)
    part_right = int(img.shape[1] * 3 / 5)
    img_part = img[part_top:, :part_right, :]
    imagePartList.append(img_part)
    # cv2.imshow("test", img_part)
    # print(img_part.shape)
    # cv2.imwrite("4.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    part_left = int(img.shape[1] * 1 / 5)
    part_right = int(img.shape[1] * 4 / 5)
    part_top = int(img.shape[0] * 1 / 4)
    img_part = img[part_top:, part_left:part_right, :]
    imagePartList.append(img_part)
    # cv2.imshow("test", img_part)
    # print(img_part.shape)
    # cv2.imwrite("5.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    part_left = int(img.shape[1] * 2 / 5)
    part_top = int(img.shape[0] * 1 / 4)
    img_part = img[part_top:, part_left:, :]
    imagePartList.append(img_part)
    # print(img_part.shape)
    # cv2.imwrite("6.jpg", img_part)
    img_part = cv2.resize(img_part, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img_part, dtype='float64') / 256
    imageList.append(img_ndarray)

    imageList = numpy.array(imageList)
    print(imageList.shape)
    imageList = imageList.reshape(imageList.shape[0], img_rows, img_cols, 3)

    result = model.predict(imageList)
    #print(result)
    result_list = []
    for result_class in result:
        # result_list.append(result_class[numpy.argmax(result_class)])
        getClass = numpy.argmax(result_class)
        if getClass == 0:
            result_list.append(result_class[getClass])
        else:
            result_list.append(0)
    print(result_list)
    result_array = numpy.asarray(result_list)
    imagePartArray = numpy.asarray(imagePartList)
    print(imagePartArray.shape)
    mostLikeId = numpy.argmax(result_array)
    print(mostLikeId)
    print(result_array[mostLikeId])

    cv2.imwrite(write_path + imgPath, imagePartArray[mostLikeId])
