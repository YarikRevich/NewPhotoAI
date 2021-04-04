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
    #Using previously trained model predicts passed photos ...

    def __init__(self, stock_model):
        self.stock_model = stock_model
        self._load_weights()

    def _load_weights(self):
        #Loads trained weights for the correct predicting ...

        self.stock_model.load_weights(config.FIT_PATH)

    def recognize(self, image_in_bytes: bytes) -> str:
        #Transorms gotten bytes into photo and predicts
        #if it is into available types ...

        image = np.fromstring(image_in_bytes, np.uint8)
        image_cv2 = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_cv2 = cv2.resize(image_cv2, (128, 128))
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        image_in_array = np.array(np.asarray(image_cv2, dtype="float").reshape(1, 128, 128, 1), dtype="float")/255.0
        image_to_check = np.expand_dims(image_in_array, axis=3)
        res = self.stock_model.predict(image_to_check)
        me = np.argmax(res)
        tags = ""
        if me == 0:
            pass
        elif me == 1:
            tags = "human;face;people;mankind"
        elif me == 2:
            tags = "animals;friends"
        elif me == 3:
            tags = "fruit;useful"
        elif me == 4:
            tags = "vehicle;transport"
        print(tags)
        return tags

        

