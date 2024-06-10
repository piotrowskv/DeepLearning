import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from diffusion_model import p_losses
from generate import *
class GenerateImagesCallback(keras.callbacks.Callback):
    def __init__(self, filepath, T):
        super().__init__()
        self.filepath= 'results/' + filepath
        self.T=T
    def on_epoch_end(self, epoch, logs=None):
        for i in range(5):
            generate_images(self.model, 1, self.filepath + str(epoch +1), self.T)

def train(model_number, train_dataset, epochs, diffusion_coefficient, T, model_name):

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
    save_model = keras.callbacks.ModelCheckpoint(
        filepath= 'results/' + model_name + '.keras',
        monitor='loss',
        mode='min',
        save_best_only=True)
    generate_images_callback = GenerateImagesCallback(model_name, T)

    unet = build_unet((64, 64, 3), model_number)
    diffusion_model = DiffusionModel(unet, timesteps=T, diffusion_coefficient=diffusion_coefficient)
    diffusion_model.compile(optimizer='adam')
    # Train the model
    diffusion_model.fit(train_dataset, epochs=epochs, callbacks=[early_stopping, save_model, generate_images_callback])
