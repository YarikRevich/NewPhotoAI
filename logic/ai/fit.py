from tensorflow.keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from logic.ai.config import DOGSET_PATH, FACESET_PATH, BIRDSET_PATH, FLOWER_PATH, FRUIT_PATH, VEHICLE_PATH, OTHERS_PATH, FIT_PATH

from typing import List, Dict, Tuple
import random
import os

class Helper:
    
    def create_random_tag(self, ot: int) -> str:
        ft = f"{ot}_"
        for i in range(9):
            ft += str(random.randint(0, 9))
        return ft

    def shuffle_dict(self, d: Dict[str, np.array]) -> Dict[str, np.array]:
        l = list(d.items())
        random.shuffle(l)
        return dict(l)

    def unparse_data_dict(self, d: Dict[str, np.array]) -> Tuple[List[np.array], List[int]]:
        labels = []
        photos = []
        for l, p in d.items():   
            labels.append(int(l.split("_")[0]))
            photos.append(p)
        return (photos, labels)
    

class DataSet(Helper):
    #Does the data preparing for the futher training ...

    def load_paths(self) -> List[str]:
        #Loads paths of datasets into the array ...

        image_paths = []
        
        for img in os.listdir(DOGSET_PATH):
            image_paths.append(f"{DOGSET_PATH}/{img}")
        for img in os.listdir(FACESET_PATH):
            image_paths.append(f"{FACESET_PATH}/{img}")
        for img in os.listdir(BIRDSET_PATH):
            image_paths.append(f"{BIRDSET_PATH}/{img}")
        for img in os.listdir(FLOWER_PATH):
            image_paths.append(f"{FLOWER_PATH}/{img}")
        for img in os.listdir(FRUIT_PATH):
            image_paths.append(f"{FRUIT_PATH}/{img}")
        for img in os.listdir(VEHICLE_PATH):
            image_paths.append(f"{VEHICLE_PATH}/{img}")
        for img in os.listdir(OTHERS_PATH):
            image_paths.append(f"{OTHERS_PATH}/{img}")
        return image_paths


    def load_photos_and_labels_by_paths(self, paths: List[str]) -> Dict[str, np.array]:
        #Due to the gotten paths gets the data for training saving equal labels ...

        d = {}
        for path in paths:
            image = cv2.imread(path)
            image = cv2.resize(image, (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(np.asarray(image, dtype="float"), dtype="float")/255.0

            if FACESET_PATH in path:
                d[self.create_random_tag(1)] = image
            elif DOGSET_PATH in path or BIRDSET_PATH in path:
                d[self.create_random_tag(2)] = image
            elif FRUIT_PATH in path:
                d[self.create_random_tag(3)] = image
            elif VEHICLE_PATH in path:
                d[self.create_random_tag(4)] = image
            else:
                d[self.create_random_tag(0)] = image

        return d

    def load_data(self) -> (List[np.array], List[np.array]):
        loaded_paths = self.load_paths()
        fr = self.load_photos_and_labels_by_paths(loaded_paths) 
        re = self.shuffle_dict(fr)
        return self.unparse_data_dict(re)


class Fit(DataSet):
    #Contains all the methods to do the fitting
    #Firstly get the data to fit over
    #Then creates the callback for weights saving
    #Afterwards does the training

    def __init__(self, stock_model):
        self.stock_model = stock_model
        self._set_data()
        self._set_cp_callback()
        self._set_fit()

    def _set_data(self):
        #Gets the data for training

        self.x_test, self.y_test = self.load_data()

        self.x_test = np.expand_dims(self.x_test, axis=3)
        
        self.y_test_array = keras.utils.to_categorical(self.y_test, 5)

    def _set_cp_callback(self):
        #Creates the callback for weights saving

        self.cp_callback = keras.callbacks.ModelCheckpoint(
            save_weights_only=True, filepath=FIT_PATH, verbose=1
        )

    def _set_fit(self):
        #Does the training ...

        self.stock_model.fit(self.x_test, self.y_test_array, batch_size=32, epochs=5,
                            validation_split=0.2, callbacks=[self.cp_callback])
