import os
import shutil


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


source_directory = 'all_images\\train'  # Your source directory containing subdirectories
target_directory = 'all_images\\train\\bed'  # Target directory where all images will be moved

move_images_to_one_folder(source_directory, target_directory)
