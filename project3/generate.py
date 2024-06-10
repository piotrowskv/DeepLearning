import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from diffusion_model import *
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model



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

def resize(image_list):
    new_image_list = []
    for img in image_list:
        new_img = tf.keras.utils.load_img(img, target_size=(64, 64, 3))
        new_image_list.append(tf.image.resize(new_img, (75, 75)))
    return new_image_list

def calculate_fid(real_images, generated_images):
    inception_model = InceptionV3(
        include_top=False, pooling='avg', input_shape=(75, 75, 3))
    
    # real_images = tf.map_fn(resize, real_images)
    # generated_images = tf.map_fn(resize, generated_images)
    real_images = resize(real_images)
    generated_images = resize(generated_images)
    # Extract features
    real_features = inception_model.predict(
        tf.data.Dataset.from_tensor_slices(real_images).batch(1))
    generated_features = inception_model.predict(
        tf.data.Dataset.from_tensor_slices(generated_images).batch(1))

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


# def calculate_fid(image1, image2):
#     def load_image(image_path):
#         img = tf.io.read_file(image_path)
#         img = tf.image.decode_image(img, channels=3)
#         # Resize to InceptionV3 input size
#         img = tf.image.resize(img, (299, 299))
#         img = tf.keras.applications.inception_v3.preprocess_input(img)
#         return img

#     def extract_features(images):
#         # Load the InceptionV3 model pre-trained on ImageNet
#         base_model = InceptionV3(
#             include_top=False, pooling='avg', input_shape=(75, 75, 3))
#         model = Model(inputs=base_model.input, outputs=base_model.output)

#         # Extract features
#         features = model.predict(images)
#         return features

#     def calculate_statistics(features):
#         mu = np.mean(features, axis=0)
#         sigma = np.cov(features, rowvar=False)
#         return mu, sigma

#     def calculate_fid2(mu1, sigma1, mu2, sigma2):
#         # Calculate the sum of the covariance matrices
#         print(sigma1)
#         print(sigma2)
#         # sigma1 += np.eye(sigma1.shape[0]) * 1e-6
#         # sigma2 += np.eye(sigma2.shape[0]) * 1e-6
#         # covmean = sqrtm(sigma1.dot(sigma2))
#         covmean = np.sqrt(sigma1 * sigma2)
#         # Numerical stability
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real

#         # Calculate the Fréchet distance
#         # fid = np.sum((mu1 - mu2) ** 2) + \
#         #     np.trace(sigma1 + sigma2 - 2 * covmean)
#         fid = np.sum((mu1 - mu2) ** 2) + \
#            (sigma1 + sigma2 - 2 * covmean)
#         print(fid)
#         return fid


#     # Create a batch of images
#     image1 = tf.image.resize(image1, (75, 75))
#     image2 = tf.image.resize(image2, (75, 75))

#     images = tf.stack([image1, image2])

#     # Extract features
#     features = extract_features(images)

#     # Compute statistics
#     mu1, sigma1 = calculate_statistics(features[:1])
#     mu2, sigma2 = calculate_statistics(features[1:])

#     # Compute FID
#     fid = calculate_fid2(mu1, sigma1, mu2, sigma2)
#     return fid
