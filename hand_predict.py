import os
import cv2
import numpy
from keras.models import load_model

model = load_model("my_transfer.h5")

dir_path = "data/train/v/"
img_rows = 224
img_cols = 224

imageList = []
imageAllFile = os.listdir(dir_path)
for imgPath in imageAllFile[:10]:
    img = cv2.imread(dir_path + imgPath)
    img = cv2.resize(img, (img_rows, img_cols))
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    if (img_ndarray.shape == (img_rows, img_cols, 3)):
        imageList.append(img_ndarray)
imageList = numpy.array(imageList)
print(imageList.shape)
imageList = imageList.reshape(imageList.shape[0], img_rows, img_cols, 3)

result = model.predict(imageList)
for result_class in result:
    print(numpy.argmax(result_class))
