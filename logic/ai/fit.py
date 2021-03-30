from tensorflow.keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

import stock_model
import config

from typing import List
import random
import os


FACESET_PATH = "dataset/test/face"
DOGSET_PATH = "dataset/test/dog"
BIRDSET_PATH = "dataset/test/bird"

class DataSet:

    def load_paths(self) -> List[str]:
        image_paths = []
        
        for img in os.listdir(DOGSET_PATH):
            image_paths.append(f"{DOGSET_PATH}/{img}")
        for img in os.listdir(FACESET_PATH):
            image_paths.append(f"{FACESET_PATH}/{img}")
        for img in os.listdir(BIRDSET_PATH):
            image_paths.append(f"{BIRDSET_PATH}/{img}")
        return image_paths


    def load_photos_by_paths(self, paths: List[str]) -> (List[np.array], List[np.array]):
        images = []
        labels = []
        for path in paths:
            image = cv2.imread(path)
            image = cv2.resize(image, (128, 128))
            image = np.asarray(image, dtype="float")
            images.append(image)
            if FACESET_PATH in path:
                labels.append(1)
            elif DOGSET_PATH in path:
                labels.append(2)
            else:
                labels.append(0)
        return (np.array(images, dtype="float")/255.0, np.array(labels))

    def load_data(self) -> (List[np.array], List[np.array]):
        loaded_paths = self.load_paths()
        return self.load_photos_by_paths(loaded_paths)


class Fit(DataSet):

    def __init__(self, stock_model):
        self.stock_model = stock_model
        self._set_data()
        self._set_cp_callback()
        self._set_fit()

    def _set_data(self):
        self.x_test, self.y_test = self.load_data()

        self.y_test_array = keras.utils.to_categorical(self.y_test, 3)

    def _set_cp_callback(self):

        self.cp_callback = keras.callbacks.ModelCheckpoint(
            save_weights_only=True, filepath=config.FIT_PATH, verbose=1
        )

    def _set_fit(self):
        self.stock_model.fit(self.x_test, self.y_test_array, batch_size=10, epochs=20,
                            validation_split=0.2, callbacks=[self.cp_callback])

if __name__ == "__main__":
    Fit(stock_model.create_stock_model())
