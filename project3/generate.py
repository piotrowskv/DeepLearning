import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from diffusion_model import *
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input



def generate_images(model, num_images, save_path, T):
    generated_images = []
    for i in range(num_images):
        noise = tf.random.normal((1, 64, 64, 3))
        for t in reversed(range(T)):
            t = tf.constant([[t /float(T)]], dtype=tf.float32)
            noise = model(noise, t)
        plt.imshow((noise[0] + 1) / 2)
        generated_images.append((noise[0] + 1) / 2)
        plt.axis('off')
        plt.savefig(save_path + '.png')
        return generated_images

def resize(img):
    return tf.image.resize(img, (299, 299))
def calculate_fid(real_images, generated_images):
    inception_model = InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_images = tf.map_fn(resize, real_images)
    generated_images = tf.map_fn(resize, generated_images)
    # Extract features
    real_features = inception_model.predict(real_images)
    generated_features = inception_model.predict(generated_images)

    # Calculate mean and covariance
    mu_real, sigma_real = real_features.mean(
        axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = generated_features.mean(
        axis=0), np.cov(generated_features, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu_real - mu_generated)**2.0)
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma_real + sigma_generated - 2.0 * covmean)
    return fid
