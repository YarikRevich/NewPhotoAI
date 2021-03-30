from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib
import sys
from logic.ai.stock_model import create_stock_model
from logic.ai import config
import cv2
import os

class FittedModel:

    def __init__(self, stock_model):
        self.stock_model = stock_model
        self._load_weights()

    def _load_weights(self):
        self.stock_model.load_weights(config.FIT_PATH)

    def is_human(self, image_in_bytes: bytes) -> bool:
        image = np.fromstring(image_in_bytes, np.uint8)
        image_cv2 = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_cv2 = cv2.resize(image_cv2, (128, 128))
        image_in_array = np.asarray(image_cv2, dtype="float")/255
        image_to_check = np.expand_dims(image_in_array, axis=0)
        res = self.stock_model.predict(image_to_check)
        if np.argmax(res) == 1:
            return True
        return False

    def is_dog(self, image_in_bytes: bytes) -> bool:
        image = np.fromstring(image_in_bytes, np.uint8)
        image_cv2 = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_cv2 = cv2.resize(image_cv2, (128, 128))
        image_in_array = np.asarray(image_cv2, dtype="float")/255
        image_to_check = np.expand_dims(image_in_array, axis=0)
        res = self.stock_model.predict(image_to_check)
        if np.argmax(res) == 2:
            return True
        return False

