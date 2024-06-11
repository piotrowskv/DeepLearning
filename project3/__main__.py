import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt
from load_data import preprocess_data, load_data
from diffusion_model import *
from train import train
from generate import *


# Load the dataset
train_dir = 'all_images'
batch_size = 64
image_size = (64, 64)
T = [2, 5, 10, 50]
diffs = [0.5, 0.6, 0.7, 0.8]
seeds = [42, 51, 128]
model_numbers = [1, 2, 3]
epochs = 2
train_dataset = load_data(
    train_dir, image_size=image_size, batch_size=batch_size)

train_dataset = preprocess_data(train_dataset)
for seed in seeds:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    for t in T:
        for diff in diffs:
            for model_num in model_numbers:
                print("\n\nTraining model number: ", model_num, "with seed: ", seed, "T: ", t, "diff: ", diff, "\n")
                model_name = "model-n" + \
                    str(model_num) + "-seed" + str(seed) + \
                    "-timesteps" + str(t) + "diff-" + str(diff)
                model = train(model_number=model_num, train_dataset=train_dataset, epochs=epochs,
                              diffusion_coefficient=diff, T=t, model_name=model_name)
                generated_images = generate_images(
                    model, 64, model_name, t)
                real_images = train_dataset.take(64)

                fid = calculate_fid(real_images, generated_images)
                print(fid)
                with open(model_name + ".txt", "w+") as f:
                    f.write(fid)
