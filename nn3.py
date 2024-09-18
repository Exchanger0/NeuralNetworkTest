from keras import layers as kl, models as km
import keras
import numpy as np
from PIL import Image
import idx2numpy


def load_emnist():
    x_train = idx2numpy.convert_from_file('emnist_dataset/letters/emnist-letters-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file('emnist_dataset/letters/emnist-letters-train-labels-idx1-ubyte')
    x_train = x_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    return x_train, y_train


# поворачивает и отзеркаливает каждое изображение
def correct_image(images_arr):
    images_arr = np.rot90(images_arr, k=3, axes=(1, 2))
    images_arr = np.flip(images_arr, axis=2)
    return images_arr


x_train, y_train = load_emnist()
x_train = correct_image(x_train)

y_train -= 1
x_train = x_train / 255

x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train, 26)

print(x_train.shape)

model = km.Sequential([
    kl.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    kl.MaxPooling2D((2, 2), 2),
    kl.Conv2D(64, (3, 3), padding="same", activation="relu"),
    kl.MaxPooling2D((2, 2), 2),
    kl.Flatten(),
    kl.Dense(132, activation="relu"),
    kl.Dense(26, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

letters = "abcdefghijklmnopqrstuvwxyz"

print("q для выхода")
path = input("Введите путь до изображения буквы:")
while path != "q":
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = 1 - img_array
    res = model.predict(img_array)
    print("Ответ нейросети:", letters[np.argmax(res)])
    path = input("Введите путь до изображения буквы:")