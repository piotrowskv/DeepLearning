import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt

def build_unet(input_shape, model_number):

    match model_number:
        case 1:
            inputs = layers.Input(shape=input_shape)

            # Downsampling
            conv1 = layers.Conv2D(32, (3, 3), activation='tanh',
                                padding='same')(inputs)
            conv1 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv1)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = layers.Conv2D(64, (3, 3), activation='tanh', padding='same')(pool1)
            conv2 = layers.Conv2D(64, (3, 3), activation='tanh', padding='same')(conv2)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = layers.Conv2D(128, (3, 3), activation='tanh',
                                padding='same')(pool2)
            conv3 = layers.Conv2D(128, (3, 3), activation='tanh',
                                padding='same')(conv3)
            pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

            # Bottleneck
            conv4 = layers.Conv2D(256, (3, 3), activation='tanh',
                                padding='same')(pool3)
            conv4 = layers.Conv2D(256, (3, 3), activation='tanh',
                                padding='same')(conv4)

            # Upsampling
            up5 = layers.Conv2D(128, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv4))
            merge5 = layers.concatenate([conv3, up5], axis=3)
            conv5 = layers.Conv2D(128, (3, 3), activation='tanh',
                                padding='same')(merge5)
            conv5 = layers.Conv2D(128, (3, 3), activation='tanh',
                                padding='same')(conv5)

            up6 = layers.Conv2D(64, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv5))
            merge6 = layers.concatenate([conv2, up6], axis=3)
            conv6 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(merge6)
            conv6 = layers.Conv2D(64, (3, 3), activation='tanh', padding='same')(conv6)

            up7 = layers.Conv2D(32, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv6))
            merge7 = layers.concatenate([conv1, up7], axis=3)
            conv7 = layers.Conv2D(32, (3, 3), activation='tanh',
                                padding='same')(merge7)
            conv7 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv7)
            conv7 = layers.Conv2D(3, (3, 3), activation='tanh',
                                padding='same')(conv7)

            model = models.Model(inputs=inputs, outputs=conv7)
        case 2:
            inputs = layers.Input(shape=input_shape)

            # Downsampling

            conv2 = layers.Conv2D(32, (3, 3), activation='linear', padding='same')(inputs)
            conv2 = layers.Conv2D(32, (3, 3), activation='linear', padding='same')(conv2)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(pool2)
            conv3 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(conv3)
            pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

            # Bottleneck
            conv4 = layers.Conv2D(128, (3, 3), activation='linear',
                                padding='same')(pool3)
            conv4 = layers.Conv2D(128, (3, 3), activation='linear',
                                padding='same')(conv4)

            # Upsampling
            up5 = layers.Conv2D(64, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv4))
            merge5 = layers.concatenate([conv3, up5], axis=3)
            conv5 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(merge5)
            conv5 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(conv5)

            up6 = layers.Conv2D(32, (2, 2), activation='linear', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv5))
            merge6 = layers.concatenate([conv2, up6], axis=3)
            conv6 = layers.Conv2D(32, (3, 3), activation='linear',
                                padding='same')(merge6)
            conv6 = layers.Conv2D(32, (3, 3), activation='linear', padding='same')(conv6)

            conv7 = layers.Conv2D(3, (3, 3), activation='tanh',
                                padding='same')(conv6)

            model = models.Model(inputs=inputs, outputs=conv7)
        case 3:
            inputs = layers.Input(shape=input_shape)

            # Downsampling
            conv1 = layers.Conv2D(32, (3, 3), activation='tanh',
                                padding='same')(inputs)
            conv1 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv1)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(pool1)
            conv2 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv2)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(pool2)
            conv3 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(conv3)
            pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

            # Bottleneck
            conv4 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(pool3)
            conv4 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(conv4)

            # Upsampling
            up5 = layers.Conv2D(64, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv4))
            merge5 = layers.concatenate([conv3, up5], axis=3)
            conv5 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(merge5)
            conv5 = layers.Conv2D(64, (3, 3), activation='tanh',
                                padding='same')(conv5)

            up6 = layers.Conv2D(32, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv5))
            merge6 = layers.concatenate([conv2, up6], axis=3)
            conv6 = layers.Conv2D(32, (3, 3), activation='tanh',
                                padding='same')(merge6)
            conv6 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv6)

            up7 = layers.Conv2D(32, (2, 2), activation='tanh', padding='same')(
                layers.UpSampling2D(size=(2, 2))(conv6))
            merge7 = layers.concatenate([conv1, up7], axis=3)
            conv7 = layers.Conv2D(32, (3, 3), activation='tanh',
                                padding='same')(merge7)
            conv7 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same')(conv7)
            conv7 = layers.Conv2D(3, (3, 3), activation='tanh',
                                padding='same')(conv7)

            model = models.Model(inputs=inputs, outputs=conv7)
    return model


class DiffusionModel(tf.keras.Model):
    def __init__(self, unet, timesteps=100, diffusion_coefficient=0.1):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.diffusion_coeff = diffusion_coefficient

    def call(self, x, t):
        noise = tf.random.normal(shape=tf.shape(x))
        return self.unet(x * (1 - t) +  self.diffusion_coeff * noise * t)

    def train_step(self, data):
        images = data
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform((batch_size, 1, 1, 1), 0, 1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self(images, t) - images)**2)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}

def q_sample(x_start, t, noise, diffusion_coefficient=0.5):
    return (1 - t) * x_start + t * diffusion_coefficient * noise


def p_losses(model, x_start, t, noise, diffusion_coefficient=0.5):
    x_noisy = q_sample(x_start, t, noise, diffusion_coefficient)
    x_recon = model(x_noisy)
    return tf.reduce_mean((x_recon - x_start) ** 2)
