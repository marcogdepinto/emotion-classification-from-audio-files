"""
File used to make a plot of the model.
"""

import keras
from keras.utils import plot_model

from config import MODEL_DIR_PATH

restored_keras_model = keras.models.load_model(MODEL_DIR_PATH + 'Emotion_Voice_Detection_Model.h5')

plot_model(restored_keras_model, to_file='media/model.png')