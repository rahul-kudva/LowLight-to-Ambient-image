import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def load_images_from_folder(folder_path, target_size=(256, 256)):
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load and resize image
            image = load_img(file_path, target_size=target_size)
            # Convert image to array
            image_array = img_to_array(image)
            images.append(image_array)
        except Exception as e:
            print(f"Could not load image {file_name}: {e}")
    return np.array(images)


if __name__ == "__main__":
    base_path = "TestDataset"  
    high_path = os.path.join(base_path, "High")  
    low_path = os.path.join(base_path, "Low")               
    
    high_images = load_images_from_folder(high_path)
    low_images = load_images_from_folder(low_path)

    
    np.savez_compressed("test_dataset.npz", high_images, low_images)
    print("Dataset saved as 'dataset.npz'.")
