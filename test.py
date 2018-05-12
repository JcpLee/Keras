import keras_test

prediction = keras_test.model.predict_classes(keras_test.X_test)
print(prediction[1])
print(prediction.shape)