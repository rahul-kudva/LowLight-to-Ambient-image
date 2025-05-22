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


# In[ ]:







# In[ ]:





# In[2]:


if tf.config.list_physical_devices('GPU'):
    print("[+] GPU is available and will be used by TensorFlow.")
    print(tf.config.list_physical_devices('GPU'))
else:
    print("[-] No GPU found. TensorFlow will use the CPU.")



def define_discriminator(img_shape):
    initializer = RandomNormal(stddev=0.02)

    src_image = Input(shape=img_shape)
    target_image = Input(shape=img_shape)

    merged_images = Concatenate()([src_image, target_image])

    # Downsampling
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(merged_images)
    d = LeakyReLU(0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)

    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)

    # Final layer with sigmoid activation
    output = Conv2D(1, (4, 4), padding='same', kernel_initializer=initializer)(d)
    patch_output = Activation('sigmoid')(output)

    model = Model([src_image, target_image], patch_output)

    # Compile model with Adam optimizer
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model





def define_generator(img_shape=(256, 256, 3)):
    initializer = RandomNormal(stddev=0.02)
    input_image = Input(shape=img_shape)

    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=initializer)(input_image)
    g = BatchNormalization()(g, training=True)
    g3 = LeakyReLU(0.2)(g)

    # Downsampling Blocks
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer)(g3)
    g = BatchNormalization()(g, training=True)
    g2 = LeakyReLU(0.2)(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer)(g2)
    g = BatchNormalization()(g, training=True)
    g1 = LeakyReLU(0.2)(g)

    # Residual Blocks
    for _ in range(6):
        res = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer)(g1)
        res = BatchNormalization()(res, training=True)
        res = LeakyReLU(0.2)(res)

        res = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer)(res)
        res = BatchNormalization()(res, training=True)

        g1 = Concatenate()([res, g1])

    # Upsampling Blocks
    g = UpSampling2D(size=(2, 2))(g1)
    g = Conv2D(128, (1, 1), kernel_initializer=initializer)(g)
    g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, g2])
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(0.2)(g)

    g = UpSampling2D(size=(2, 2))(g)
    g = Conv2D(64, (1, 1), kernel_initializer=initializer)(g)
    g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, g3])
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(0.2)(g)

    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=initializer)(g)
    g = BatchNormalization()(g, training=True)
    output_image = Activation('tanh')(g)

    model = Model(input_image, output_image)
    return model




def define_gan(generator_model, discriminator_model, img_shape):
    # Freeze discriminator weights except BatchNormalization
    for layer in discriminator_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    src_input = Input(shape=img_shape)
    generated_output = generator_model(src_input)
    disc_output = discriminator_model([src_input, generated_output])
    model = Model(src_input, [disc_output, generated_output])

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100])
    return model





def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X2, X1]





def generate_real_samples(dataset, n_samples, patch_shape):
	trainA, trainB = dataset
	ix = randint(0, trainA.shape[0], n_samples)
	X1, X2 = trainA[ix], trainB[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y





def generate_fake_samples(g_model, samples, patch_shape):
	X = g_model.predict(samples)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y




def summarize_performance(step, g_model, d_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    plt.figure(figsize=(14, 14))
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.title('Low-Light')
        plt.imshow(X_realA[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.title('Generated')
        plt.imshow(X_fakeB[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.title('Ground Truth')
        plt.imshow(X_realB[i])
    filename1 = step_output + 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    filename2 = model_output + 'gen_model_%06d.h5' % (step+1)
    g_model.save(filename2)
    filename3 = model_output + 'disc_model_%06d.h5' % (step+1)
    d_model.save(filename3)
    print('[.] Saved Step : %s' % (filename1))
    print('[.] Saved Model: %s' % (filename2))
    print('[.] Saved Model: %s' % (filename3))




def train(d_model, g_model, gan_model, dataset, n_epochs=40, n_batch=12):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    print("[!] Number of steps {}".format(n_steps))
    print("[!] Saves model/step output at every {}".format(bat_per_epo * 10))
    for i in range(n_steps):
        start = time.time()
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        time_taken = time.time() - start
        print(
            '[*] %06d, d1[%.3f] d2[%.3f] g[%06.3f] ---> time[%.2f], time_left[%.08s]'
                %
            (i+1, d_loss1, d_loss2, g_loss, time_taken, str(datetime.timedelta(seconds=((time_taken) * (n_steps - (i + 1))))).split('.')[0].zfill(8))
        )
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, d_model, dataset)



dataset = load_real_samples('dataset.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)


def resize_images(dataset, new_size=(32, 32)):
    trainA, trainB = dataset
    trainA_resized = []
    trainB_resized = []
    
    for img in trainA:
        img = tf.image.resize(img, new_size)
        trainA_resized.append(img.numpy())
    
    for img in trainB:
        img = tf.image.resize(img, new_size)
        trainB_resized.append(img.numpy())
    
    return np.array(trainA_resized), np.array(trainB_resized)

dataset = resize_images(dataset, new_size=(128, 128))
image_shape = dataset[0].shape[1:]
print('Loaded', dataset[0].shape, dataset[1].shape)





d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)





fileName = 'Enhancement Model'
step_output = fileName + "/Step Output/"
model_output = fileName + "/Model Output/"
os.mkdir(fileName)
os.mkdir(step_output)
os.mkdir(model_output)

train(d_model, g_model, gan_model, dataset,80, 4)


#Testing




def normalize_image(img):
    return (img / 127.5) - 1

def denormalize_image(img):
    return np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)


# In[43]:


generator_model_path = "Enhancement Model\Model Output\gen_model_009680.h5"
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

average_psnr = total_psnr / file_count if file_count > 0 else 0

print(f"\nProcessed {file_count} files.")
print(f"Average PSNR: {average_psnr:.2f} dB")


# In[ ]:




