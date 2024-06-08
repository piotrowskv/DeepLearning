import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import keras
import matplotlib.pyplot as plt


def move_images_to_one_folder(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add more extensions if needed
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # Handle naming conflicts by appending a number to the file name
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(
                            target_dir, f"{base}_{counter}{ext}")
                        counter += 1

                shutil.move(source_path, target_path)
                print(f"Moved {source_path} to {target_path}")

def load_data(directory, image_size=(64, 64), batch_size=32):
    dataset = image_dataset_from_directory(
        directory,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size
    )
    return dataset


def preprocess_data(dataset):
    dataset = dataset.map(lambda x, y: (x - 127.5) / 127.5,
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# MOVE IMAGES DRIVER CODE
# source_directory = 'all_images\\train'
# target_directory = 'all_images\\train\\bed'

# move_images_to_one_folder(source_directory, target_directory)
