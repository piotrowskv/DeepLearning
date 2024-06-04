import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def generate_images(model, num_images, timesteps):
    for i in range(num_images):
        noise = tf.random.normal((1, 64, 64, 3))
        for t in reversed(range(int(timesteps))):
            t = tf.constant([[t / timesteps]], dtype=tf.float32)
            noise = model(noise, t)
        plt.imshow((noise[0] + 1) / 2)
        plt.axis('off')
        plt.show()


