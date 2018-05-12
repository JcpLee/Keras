import keras
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
#建立网络模型
model = Sequential()

model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))

# print(model.summary())
#配置参数
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
train_history = model.fit(x=X_train,y=y_train_label,validation_split=0.2,epochs=10,batch_size=200,verbose=2)
#画图函数
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
#测试
scores = model.evaluate(X_test,y_test_label1)
print('accracy:',scores[1])

prediction = model.predict_classes(X_test)
print(prediction[1])
# print(prediction.shape)
# pd.crosstab(y_test_label1,prediction,rownames=['label'],colnames=['predict'])