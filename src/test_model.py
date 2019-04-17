import numpy as np
import cv2
from PIL import Image
import pytesseract
import os
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.model_selection import train_test_split
from pandas import read_csv

model = load_model("../model/mnist_DNN.h5")
model.summary()

#data = np.loadtxt("../mnist.csv", skiprows=1, dtype='int', delimiter=',')
data = read_csv("../small_mnist.csv")

X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

print(X[0:5])
print(Y[0:5])

X = X.values # pandas dataframes to numpy array
X = X.reshape(-1, 28, 28, 1)
X = X.astype("float32")/255

print(X.shape)

print(model.predict_classes(X[:100]))
print(Y[0:100])

#---

def show(img, window_title):
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def resize(img, width):
    ratio = img.shape[1] / width
    width = int(img.shape[1] / ratio)
    height = int(img.shape[0] / ratio)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

img = cv2.imread("../img/seven.jpg", 0)
# img = resize(img, 600)
ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show(img, "image")

for x in range(len(img)):
    for y in range(len(img[x])):
        if img[x][y] == 0:
            img[x][y] = 255
        else:
            img[x][y] = 0

show(img, "image")

print(img.shape)

img = img.reshape(-1, 28, 28, 1)
print(img.shape)

print(model.predict_classes(img))
