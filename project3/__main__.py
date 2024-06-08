import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from load_data import preprocess_data, load_data
from diffusion_model import *
from train import train


# Load the dataset
train_dir = 'less_images'
batch_size = 64
image_size = (64, 64)
T = [2, 50, 100]
diffs = [0.1, 0.5, 1]
seeds = [42, 51, 128, 345, 11]
epochs = 3
train_dataset = load_data(
    train_dir, image_size=image_size, batch_size=batch_size)

train_dataset = preprocess_data(train_dataset)
for seed in seeds:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    for t in T:
        for diff in diffs:
            model_name = "model-seed" + str(seed) + "-timesteps" + str(t) + "diff-" + str(diff)
            train(model_number=1, train_dataset=train_dataset, epochs=epochs,
                  diffusion_coefficient=diff, T=t, model_name=model_name)
