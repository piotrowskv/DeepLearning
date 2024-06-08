import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from diffusion_model import *


def generate_images(model, num_images, save_path, T):
    for i in range(num_images):
        noise = tf.random.normal((1, 64, 64, 3))
        for t in reversed(range(T)):
            t = tf.constant([[t /float(T)]], dtype=tf.float32)
            noise = model(noise, t)
        plt.imshow((noise[0] + 1) / 2)
        plt.axis('off')
        plt.savefig(save_path + '.png')

