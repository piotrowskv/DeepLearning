import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def down_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    c = layers.Conv2D(filters, kernel_size,
                      padding=padding, strides=strides)(x)
    c = layers.BatchNormalization()(c)
    c = layers.LeakyReLU()(c)
    c = layers.Conv2D(filters, kernel_size,
                      padding=padding, strides=strides)(c)
    c = layers.BatchNormalization()(c)
    c = layers.LeakyReLU()(c)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding='same', strides=1):
    us = layers.UpSampling2D((2, 2))(x)
    concat = layers.Concatenate()([us, skip])
    c = layers.Conv2D(filters, kernel_size, padding=padding,
                      strides=strides)(concat)
    c = layers.BatchNormalization()(c)
    c = layers.LeakyReLU()(c)
    c = layers.Conv2D(filters, kernel_size,
                      padding=padding, strides=strides)(c)
    c = layers.BatchNormalization()(c)
    c = layers.LeakyReLU()(c)
    return c


def UNet(input_shape):
    f = [32, 64, 128, 256]

    inputs = layers.Input(input_shape)

    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    # c4, p4 = down_block(p3, f[3])

    bn = layers.Conv2D(f[3], (3, 3), padding='same', strides=1)(p3)
    bn = layers.BatchNormalization()(bn)
    bn = layers.LeakyReLU()(bn)

    u1 = up_block(bn, c3, f[2])
    u2 = up_block(u1, c2, f[1])
    u3 = up_block(u2, c1, f[0])

    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='tanh')(u3)

    model = tf.keras.models.Model(inputs, outputs)
    return model


class DiffusionModel(tf.keras.Model):
    def __init__(self, unet, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def call(self, x, t):
        noise = tf.random.normal(shape=tf.shape(x))
        return self.unet(x * (1 - t) + noise * t)

    def train_step(self, data):
        images = data
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform((batch_size, 1, 1, 1), 0, 1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self(images, t) - images)**2)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}
