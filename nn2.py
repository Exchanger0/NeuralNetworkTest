import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras as k
from keras.api.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

y_train = k.utils.to_categorical(y_train, 10)
y_test = k.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)

model = k.Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), 2),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2), 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Обучение нейронной сети началось")
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
print("Обучение нейронной сети закончилось")

print("q для выхода")
path = input("Введите путь до изображения цифры:")
while path != "q":
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = 1 - img_array
    res = model.predict(img_array)
    print("Ответ нейросети:", np.argmax(res))
    path = input("Введите путь до изображения цифры:")