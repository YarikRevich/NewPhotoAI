from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D


def create_stock_model():
    #Creates stock model for the futher training or just prod using ...

    model = keras.Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(5, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model
