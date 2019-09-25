import os
import os.path
import cv2
import pickle
import numpy as np
from imutils import paths
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import img_to_array
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from preprocess import resize_symbol
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


input_images_folder = "single symbols"
model_file = "model.hdf5"
labels_file = "labels.dat"

data = []
labels = []

for input_image in paths.list_images(input_images_folder):
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_symbol(image, 28)
    image = img_to_array(image)
    data.append(image)
    label = input_image.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

with open(labels_file, "wb") as f:
    pickle.dump(lb, f)
    
model = Sequential()

model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(50, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500, activation="relu"))

model.add(Dense(20, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [TensorBoard(log_dir='tb_logs', histogram_freq=1, write_images=True)]

history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10, callbacks=callbacks)

plt.plot(history.history['acc'], label='Аккуратность на обучающем наборе')
plt.plot(history.history['val_acc'], label='Аккуратность на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Аккуратность')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

model.save(model_file)































