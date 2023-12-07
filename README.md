# GAN_Anime-Face-Generation
Anime Face Generation using DCGAN

Generative Model Choice:

For generating realistic and diverse anime faces, Deep Convolutional Generative Adversarial Networks (DCGANs) are a compelling choice. 
DCGANs excel at image generation tasks due to their ability to capture complex spatial relationships and intricate details present in the training data. 
This makes them ideal for generating anime faces, which often possess unique features and stylistic elements compared to real-world faces.

The objective of the project is to Generate Anime Faces using Deep Convolutional Generative Adversarial Network (DCGAN) with Keras and Tensorflow.

Two models, namely generator and discriminator, have been trained. The Discriminator try to classify the images whether it is real (or) fake images. 
The Generator try to produce images that are close to real and fool the discriminator. Finally, the generator will produce similar images to the train dataset with different variety as output.

Libraries
numpy
matplotlib
keras
tensorflow
nltk

Neural Network
Deep Convolutional Generative Adversarial Network (DCGAN)

Objective:

This project aims to generate realistic and diverse anime faces using a Deep Convolutional Generative Adversarial Network (DCGAN) implemented with Keras and TensorFlow.

Model Choice:

DCGANs were chosen for this task due to their remarkable ability to capture intricate details and spatial relationships present in images, making them ideal for replicating the stylistic features of anime faces. Additionally, DCGANs offer:

High-quality image generation: Convolutional layers effectively capture the nuances and details unique to anime faces, leading to realistic and visually appealing outputs.
Stable training: DCGANs utilize established techniques like strided convolutions and batch normalization, promoting efficient and stable training processes.
Controllability: Conditional DCGANs allow influencing the generated content by incorporating additional information like specific hair styles, eye colors, or facial expressions.


Dataset:

A large and diverse dataset of anime faces is crucial for the DCGAN to learn the intricate details and variations characteristic of this style.

Environment: Kaggle

Download link: https://www.kaggle.com/datasets/soumikrakshit/anime-faces  

Model Architecture:

The DCGAN architecture consists of two key components:

Generator: Responsible for creating new anime faces by sampling from a latent space and progressively upsampling the features to generate realistic images.
Discriminator: Aims to distinguish between real anime faces and the generated ones, helping the generator improve its output over time.
Both networks utilize convolutional layers with appropriate filters and kernels, activation functions like Leaky ReLU, and batch normalization layers for improved training stability.

Training Process:

Data pre-processing: Images are resized, normalized, and potentially augmented to improve training efficiency and data diversity.
Hyperparameter tuning: Learning rate, batch size, and network architecture parameters are optimized for optimal performance.
Adversarial training: Both the generator and discriminator are trained simultaneously in an adversarial manner. The generator tries to fool the discriminator by creating increasingly realistic faces, while the discriminator strives to accurately identify real and fake images. This competitive process drives both networks to improve, ultimately leading to high-quality anime face generation.
Monitoring and visualization: Training losses and sample images are monitored periodically to track progress and identify any potential issues like mode collapse.


Steps to Run the Code: 

You need to set up a new environment on your computer. However, it is not compulsory to use your local machine, you can train a model on, let's say Google Colab and download the trained model to server the requests for classification.

Install required packages 

Once the virtual environment is activated run the following command to get the required packages

# Import Modules

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

warnings.filterwarnings('ignore')

Create a dataset for DCGAN

Download the images from the Kaggle Dataset link and upload it to the Google Drive. Or else directly upload the Kaggle dataset zip file to Google drive and unzip the file through Google Colab

import zipfile

zip_ref = zipfile.ZipFile("/content/drive/MyDrive/Colab Notebooks/Betterzilla_Assignment/dataset/Anime Faces.zip", 'r')
zip_ref.extractall("/content/drive/MyDrive/Colab Notebooks/Betterzilla_Assignment")
zip_ref.close()
# Load the files

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Betterzilla_Assignment/dataset/data"

# load complete image paths to the list
image_paths = []
for image_name in os.listdir(BASE_DIR):
    image_path = os.path.join(BASE_DIR, image_name)
    image_paths.append(image_path)

image_paths[:5]

# Checks the total number of images
len(image_paths)


# Visualize the Image Dataset
Displays a sample of Anime faces images
# to display grid of images (7x7)

plt.figure(figsize=(20, 20))
temp_images = image_paths[:49]
index = 1

for image_path in temp_images:
    plt.subplot(7, 7, index)
    # load the image
    img = load_img(image_path)
    # convert to numpy array
    img = np.array(img)
    # show the image
    plt.imshow(img)
    plt.axis('off')
    # increment the index for next image
    index += 1

 
 # Preprocess Images

# load the image and convert to numpy array
train_images = [np.array(load_img(path)) for path in tqdm(image_paths)]
train_images = np.array(train_images)

train_images[0].shape

# reshape the array
train_images = train_images.reshape(train_images.shape[0], 64, 64, 3).astype('float32')

# normalize the images
train_images = (train_images - 127.5) / 127.5

train_images[0]

Create Generator & Discriminator Models
Generator: Responsible for creating new anime faces by sampling from a latent space and progressively upsampling the features to generate realistic images.
Discriminator: Aims to distinguish between real anime faces and the generated ones, helping the generator improve its output over time.

# latent dimension for random noise
LATENT_DIM = 100
# weight initializer
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# no. of channels of the image
CHANNELS = 3 # for gray scale, keep it as 1

Generator Model

Generator Model will create new images similar to training data from random noise

model = Sequential(name='generator')

# 1d random noise
model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
# model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# convert 1d to 3d
model.add(layers.Reshape((8, 8, 512)))

# upsample to 16x16
model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
# model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# upsample to 32x32
model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
# model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# upsample to 64x64
model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
# model.add(layers.BatchNormalization())
model.add(layers.ReLU())

model.add(layers.Conv2D(CHANNELS, (4, 4), padding='same', activation='tanh'))

generator = model
generator.summary()

Discriminator Model

Discriminator model will classify the image from the generator to check whether it real (or) fake images.

model = Sequential(name='discriminator')
input_shape = (64, 64, 3)
alpha = 0.2

# create conv layers
model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))

# output class
model.add(layers.Dense(1, activation='sigmoid'))

discriminator = model
discriminator.summary()


Create DCGAN

class DCGAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')

    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        # get batch size from the data
        batch_size = tf.shape(real_images)[0]
        # generate random noise
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # train the discriminator with real (1) and fake (0) images
        with tf.GradientTape() as tape:
            # compute loss on real images
            pred_real = self.discriminator(real_images, training=True)
            # generate real image labels
            real_labels = tf.ones((batch_size, 1))
            # label smoothing
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            d_loss_real = self.loss_fn(real_labels, pred_real)

            # compute loss on fake images
            fake_images = self.generator(random_noise)
            pred_fake = self.discriminator(fake_images, training=True)
            # generate fake labels
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)

            # total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2

        # compute discriminator gradients
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # update the gradients
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))


        # train the generator model
        labels = tf.ones((batch_size, 1))
        # generator want discriminator to think that fake images are real
        with tf.GradientTape() as tape:
            # generate fake images from generator
            fake_images = self.generator(random_noise, training=True)
            # classify images as real or fake
            pred_fake = self.discriminator(fake_images, training=True)
            # compute loss
            g_loss = self.loss_fn(labels, pred_fake)

        # compute gradients
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        # update the gradients
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # update states for both models
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {'d_loss': self.d_loss_metric.result(), 'g_loss': self.g_loss_metric.result()}

  class DCGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_imgs=25, latent_dim=100):
        self.num_imgs = num_imgs
        self.latent_dim = latent_dim
        # create random noise for generating images
        self.noise = tf.random.normal([25, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        # generate the image from noise
        g_img = self.model.generator(self.noise)
        # denormalize the image
        g_img = (g_img * 127.5) + 127.5
        g_img.numpy()

        fig = plt.figure(figsize=(8, 8))
        for i in range(self.num_imgs):
            plt.subplot(5, 5, i+1)
            img = array_to_img(g_img[i])
            plt.imshow(img)
            plt.axis('off')
        # plt.savefig('epoch_{:03d}.png'.format(epoch))
        plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save('generator.h5')

dcgan = DCGAN(generator=generator, discriminator=discriminator, latent_dim=LATENT_DIM)

D_LR = 0.0001
G_LR = 0.0003
dcgan.compile(g_optimizer=Adam(learning_rate=G_LR, beta_1=0.5), d_optimizer=Adam(learning_rate=D_LR, beta_1=0.5), loss_fn=BinaryCrossentropy())

N_EPOCHS = 50
dcgan.fit(train_images, epochs=N_EPOCHS, callbacks=[DCGANMonitor()])

Generate New Anime Image

Now that the DC GAN model is ready, we can generate new images of Anime faces

noise = tf.random.normal([1, 100])
fig = plt.figure(figsize=(3, 3))
# generate the image from noise
g_img = dcgan.generator(noise)
# denormalize the image
g_img = (g_img * 127.5) + 127.5
g_img.numpy()
img = array_to_img(g_img[0])
plt.imshow(img)
plt.axis('off')
# plt.savefig('epoch_{:03d}.png'.format(epoch))
plt.show()

Challenges faced during implementation:
Data Acquisition and Preprocessing:

Finding a large and diverse dataset of anime faces with proper licensing and labeling information.
Balancing the dataset to ensure a representative distribution of different facial features, expressions, and styles.
Preprocessing the data efficiently to resize, normalize, and potentially augment the images for training.
Model Architecture and Hyperparameter Tuning:

Choosing the appropriate network architecture and hyperparameters (learning rate, batch size, etc.) that balance image quality, training stability, and computational efficiency.
Avoiding mode collapse, where the model generates repetitive outputs due to insufficient data diversity or suboptimal training.
Tuning the balance between the generator and discriminator to ensure a competitive training process and prevent either network from dominating the other.
Training process and Monitoring:

Ensuring stable training with minimal convergence issues or gradient vanishing/exploding problems.
Effectively monitoring the training progress through loss curves, sample images, and quantitative metrics like Inception Score and FID.
Debugging and addressing any training errors or unexpected behavior that might impact the model's performance.
Evaluation and Control:

Designing a comprehensive evaluation framework that considers both quantitative metrics and qualitative human assessment.
Achieving control over the generated content by incorporating additional information (e.g., conditioning vectors) into the model.
Balancing the balance between realism and stylistic diversity to ensure the generated faces remain both realistic and aesthetically pleasing.
