import keras
import h5py
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import pandas as pd
#数据预处理
(X_train_image,y_train_label),(X_test_image,y_test_label) = mnist.load_data()

X_train = X_train_image.reshape(60000,784).astype('float32')
X_test = X_test_image.reshape(10000,784).astype('float32')

X_train = X_train/255
X_test = X_test/255

y_train_label = np_utils.to_categorical(y_train_label)
y_test_label1 = np_utils.to_categorical(y_test_label)
from keras.models import load_model

# 从文件 my_model.h5 中载入模型
model = load_model('my_model.h5')

prediction = model.predict_classes(X_test[10].reshape(1,784))
pre = model.predict(X_test)
print(y_test_label[10])
print(prediction)
