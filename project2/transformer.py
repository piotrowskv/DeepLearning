# https://keras.io/examples/audio/transformer_asr/

import tensorflow as tf
import keras
from keras.initializers import GlorotNormal
from keras import layers

class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        initializer = GlorotNormal()
        self.conv1 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="leaky_relu", kernel_initializer=initializer
        )
        # self.conv2 = keras.layers.Conv1D(
        #     num_hid, 11, strides=2, padding="same", activation="leaky_relu"
        # )
        # self.conv3 = keras.layers.Conv1D(
        #     num_hid, 11, strides=2, padding="same", activation="leaky_relu", kernel_initializer=initializer
        # )

    def call(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        return x

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        initializer = GlorotNormal()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="leaky_relu",
                             kernel_initializer=initializer),
                layers.Dense(embed_dim, kernel_initializer=initializer),
            ]
        )
        self.layernorm1 = layers.BatchNormalization()
        self.layernorm2 = layers.BatchNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

