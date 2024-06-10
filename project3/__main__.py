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
import os

# Load the dataset
train_dir = 'all_images'
batch_size = 64
image_size = (64, 64)
T = [2, 50, 100]
diffs = [0.5]
seeds = [42, 51, 128, 345, 11]
model_numbers = [3]
epochs = 3
# train_dataset = load_data(
#     train_dir, image_size=image_size, batch_size=batch_size)

# train_dataset = preprocess_data(train_dataset)
# for seed in seeds:
#     tf.random.set_seed(seed)
#     np.random.seed(seed)

    # for t in T:
    #     for diff in diffs:
    #         for model_num in model_numbers:
    #             model_name = "2model-n" + \
    #                 str(model_num) + "-seed" + str(seed) + \
    #                 "-timesteps" + str(t) + "diff-" + str(diff)
    #             model = train(model_number=model_num, train_dataset=train_dataset, epochs=epochs,
    #                           diffusion_coefficient=diff, T=t, model_name=model_name)
#model = keras.saving.load_model("2model-n1-seed42-timesteps2diff-0.5.keras")
real_images = []
for root, _, files in os.walk('less_images'):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add more extensions if needed
            real_images.append(os.path.join(root, file))

generated_images = [
    r"model-n2-seed42-timesteps2diff-0.61.png",
    r"model-n2-seed42-timesteps2diff-0.62.png",
    r"model-n2-seed42-timesteps2diff-0.63.png",

]
fid = calculate_fid(real_images, generated_images)
print(fid)
# with open(model_name + ".txt", "w+") as f:
#     f.write(fid)
