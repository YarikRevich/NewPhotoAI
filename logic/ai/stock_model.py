from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense


def create_stock_model():

    model = keras.Sequential([
        Flatten(input_shape=(128, 128, 3)),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model
