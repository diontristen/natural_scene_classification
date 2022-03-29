import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import tensorflow as tf
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec


TEST_PATH = './testset'
model = tf.keras.models.load_model('models/model_1.h5')  # 2 dense layer 2 convo layer 128 nodes


def get_classlabel(class_code):
    labels = {2: 'glacier', 4: 'sea', 0: 'buildings', 1: 'forest', 5: 'street', 3: 'mountain'}
    return labels[class_code]

def prepare_image(number):
    image = cv2.imread(f"{TEST_PATH}/{number}.jpeg")
    pred_images = cv2.resize(image, (150, 150))
    pred_images = np.array([pred_images])
    return pred_images, image


test_number = 1
while os.path.isfile(f"{TEST_PATH}/{test_number}.jpeg"):
    try:
        image, originalImage = prepare_image(test_number)
        prediction = model.predict(image)
        idx_prediction = np.argmax(prediction[0])
        klass = get_classlabel(idx_prediction)
        plot.imshow(originalImage)
        plot.title(f"Testing Number: {test_number} \n Prediction: {klass}")
        plot.show()
    except Exception as e:
        print("Something went wrong with the image...")
        print(e)
    finally:
        test_number += 1






