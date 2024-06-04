import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpltimg
import tensorflow_datasets as tfds
from diffusion_model import UNet, DiffusionModel
import os
from generate import generate_images
def preprocess_image(image, label):
    image = tf.image.resize(image, (64, 64))
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]

    return image

if __name__ == "__main__":
    builder = tfds.ImageFolder(os.path.join(os.getcwd(), "all_images"))
   # print(os.path.join(os.getcwd(), "all_images"))
    print(builder.info)
    dataset = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
    #dataset, info = tfds.load('lsun/bedroom', split='train', with_info=True)
    dataset = dataset.map(preprocess_image).batch(4).shuffle(100)
    unet = UNet((64, 64, 3))
    diffusion_model = DiffusionModel(unet, timesteps=5.0)
    diffusion_model.compile(optimizer='adam')

    # Train the model
    diffusion_model.fit(dataset, epochs=5)
    generate_images(diffusion_model, 3, timesteps=5.0)
