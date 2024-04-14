import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = 'data' 
AUGMENTED_DIR = os.path.join(DATA_DIR, 'train-augmented')

def rotate_image(image):
    angle = np.random.randint(-30, 31)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale=1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def scale_image(image):
    fx = fy = np.random.uniform(1.1, 1.3)
    scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    start_row = int((scaled_image.shape[0] - image.shape[0]) / 2)
    start_col = int((scaled_image.shape[1] - image.shape[1]) / 2)
    cropped_image = scaled_image[start_row:start_row + image.shape[0], start_col:start_col + image.shape[1]]
    return cropped_image

def add_noise_to_image(image, mean=0, sigma=25):
    noise = np.random.normal(loc=mean, scale=sigma, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def create_augmented_folder(augmented_dir):
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

def add_augmentation_name(original_name, augmentation_suffix):
    name_part, extension_part = os.path.splitext(original_name)
    return f"{name_part}_{augmentation_suffix}{extension_part}"

def augment_data(base_dir, augmented_dir):
    create_augmented_folder(augmented_dir)
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        augmented_class_dir = os.path.join(augmented_dir, class_name)
        if not os.path.exists(augmented_class_dir):
            os.makedirs(augmented_class_dir)
        if os.path.isdir(class_dir):
            image_names = os.listdir(class_dir)
            # Wrap the loop with tqdm for a progress bar
            for image_name in tqdm(image_names, desc=f"Processing {class_name}"):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Original image
                    cv2.imwrite(os.path.join(augmented_class_dir, image_name), image)

                    # Individual augmentations
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "rotated")), rotate_image(image))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "scaled")), scale_image(image))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "noisy")), add_noise_to_image(image))
                    
                    # Combinations
                    rotated_scaled = scale_image(rotate_image(image))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "rotated_scaled")), rotated_scaled)

                    rotated_noisy = add_noise_to_image(rotate_image(image))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "rotated_noisy")), rotated_noisy)

                    scaled_noisy = add_noise_to_image(scale_image(image))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "scaled_noisy")), scaled_noisy)

                    all_augmented = add_noise_to_image(scale_image(rotate_image(image)))
                    cv2.imwrite(os.path.join(augmented_class_dir, add_augmentation_name(image_name, "all_augmented")), all_augmented)

if __name__ == '__main__':
    augment_data(os.path.join(DATA_DIR, 'train'), AUGMENTED_DIR)
