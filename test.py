#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import time
import datetime
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np
from PIL import Image
from tensorflow.keras import Input
from numpy import load, zeros, ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, ELU, Activation
from tensorflow.keras.layers import Concatenate, Dropout, BatchNormalization, LeakyReLU, Add, UpSampling2D
from keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



def normalize_image(img):
    return (img / 127.5) - 1

# Function to denormalize an image to [0, 255] range
def denormalize_image(img):
    return np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)


# In[43]:


generator_model_path = "gen_model_009680.h5"
output_directory = "Output_images"
os.makedirs(output_directory, exist_ok=True)

# Load generator model
generator = load_model(generator_model_path)

total_psnr = 0
file_count = 0

file_path = "test_dataset.npz"
data = np.load(file_path)

low_light_images = data['arr_1'] 
target_images = data['arr_0']   

assert low_light_images.shape[0] == target_images.shape[0], "Mismatch in dataset size!"

for idx in range(low_light_images.shape[0]):
    low_light_image = low_light_images[idx]
    target_image = target_images[idx]

    low_light_image_resized = cv2.resize(low_light_image, (128, 128))
    target_image_resized = cv2.resize(target_image, (128, 128))

    low_light_image_normalized = normalize_image(low_light_image_resized)
    low_light_image_normalized = np.expand_dims(low_light_image_normalized, axis=0)  # Add batch dimension

    enhanced_image_normalized = generator.predict(low_light_image_normalized)[0]  # Remove batch dimension
    enhanced_image = denormalize_image(enhanced_image_normalized)

    psnr_value = psnr(target_image_resized, enhanced_image, data_range=255)

    total_psnr += psnr_value
    file_count += 1

    output_image_path = os.path.join(output_directory, f"enhanced_{idx}.jpg")
    enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, enhanced_image_bgr)

    print(f"Processed image {idx + 1} - PSNR: {psnr_value:.2f}")






