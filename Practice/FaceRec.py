import functools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tqdm import tqdm

assert len(tf.config.list_physical_devices('GPU')) > 0

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

num_epochs = 5
num_filters = 36
batch_size = 16
learning_rate = 2e-3
latent_dim = 20


def plot_sample(x, vae):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)

    idx = random.randint(0, batch_size - 1)  # np.where(y==1)[0][0]
    plt.imshow(np.squeeze(x[idx]))
    plt.grid(False)

    plt.subplot(1, 2, 2)
    recon, _ = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(np.squeeze(recon[idx]))
    plt.grid(False)

    plt.show()


def vae(latent_dim=40):
    dense = functools.partial(tf.keras.layers.Dense, activation='relu')
    conv2D = tf.keras.layers.Conv2D
    conv2DT = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    reshape = tf.keras.layers.Reshape
    normalize = tf.keras.layers.BatchNormalization
    flatten = tf.keras.layers.Flatten

    encoder = tf.keras.Sequential([
        conv2D(num_filters, (3, 3), strides=2, activation='softplus'),
        normalize(),
        conv2D(num_filters, (3, 3), strides=2, activation='softplus'),
        normalize(),
        flatten(),
        # dense(128),
        dense(latent_dim, activation=None),
    ])

    decoder = tf.keras.Sequential([
        dense(num_filters * 7 * 7),
        reshape(target_shape=(7, 7, num_filters)),
        conv2DT(num_filters, (3, 3), strides=2, activation='softplus'),
        normalize(),
        conv2DT(1, (3, 3), strides=2, activation='softplus'),
    ])

    return encoder, decoder


class FaceRec(tf.keras.Model):
    def __init__(self, latent_dim, optimizer=tf.keras.optimizers.Adamax(5e-3)):
        super(FaceRec, self).__init__()
        self.latent_dim = latent_dim
        self.encoder, self.decoder = vae(latent_dim)
        self.encoder.compile(optimizer=optimizer)
        self.decoder.compile(optimizer=optimizer)

    def encode(self, x):
        # encoder output
        # encoder_output = self.encoder(x)
        # classification prediction
        z = self.encoder(x)
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # Encode input to a prediction and latent space
        z = self.encode(x)
        # reconstruction
        recon = self.decode(z)
        return recon, z

    def save(self, fname):
        dir = os.getcwd() + '/'
        if not os.path.isdir(dir + fname):
            try:
                os.mkdir(dir + fname)
            except OSError:
                print("File save failed.")
        self.encoder.save(dir + fname + '/encoder')
        self.decoder.save(dir + fname + '/decoder')

    def load(self, fname):
        dir = os.getcwd() + '/'
        self.encoder = tf.keras.models.load_model(dir + fname + '/encoder')
        self.decoder = tf.keras.models.load_model(dir + fname + '/decoder')


def train_step(x, fRec, optimizer):
    with tf.GradientTape() as tape:
        x_recon, z = fRec(x)
        loss = tf.reduce_mean(tf.abs(x - x_recon), axis=(1, 2, 3))
    grads = tape.gradient(loss, fRec.trainable_variables)
    optimizer.apply_gradients(zip(grads, fRec.trainable_variables))
    return loss


def train_model(latent_dim=40, fname='MODEL', optimizer=tf.keras.optimizers.Adamax(5e-3)):
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = (np.expand_dims(train_images, axis=-1) / 255.).astype(np.float32)
    train_labels = (train_labels).astype(np.int64)
    test_images = (np.expand_dims(test_images, axis=-1) / 255.).astype(np.float32)
    test_labels = (test_labels).astype(np.int64)

    fRec = FaceRec(latent_dim=latent_dim)
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists

    for i in range(num_epochs):
        print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
        for idx in tqdm(range(0, train_images.shape[0], batch_size)):
            # First grab a batch of training data and convert the input images to tensors
            (images, labels) = (train_images[idx:idx + batch_size], train_labels[idx:idx + batch_size])
            images = tf.convert_to_tensor(images, dtype=tf.float32)
            # print(tf.shape(images[0]))
            loss = train_step(images, fRec, optimizer)
            if idx % 30000 == 0:
                plot_sample(images, fRec)

    fRec.save(fname)


def load_model(latent_dim, fname):
    fRec = FaceRec(latent_dim)
    fRec.load(fname)
    return fRec


optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
# train_model(latent_dim=latent_dim, fname='MNIST_rec', optimizer=optimizer)

# ''' Testing the model by inputting arbitrary inputs or by modifying values of the compressed format

fRec = load_model(latent_dim=latent_dim, fname='MNIST_rec')

# image = mpimg.imread(os.getcwd()+'/test.png')

guyo = tf.convert_to_tensor([tf.abs(tf.random.normal((28, 28, 1), mean=0.5, stddev=0.3))])

plt.figure(figsize=(6, 6))
plt.subplot(3, 2, 1)

plt.imshow(np.squeeze(guyo))
plt.grid(False)

plt.subplot(3, 2, 2)
recon, z = fRec.predict(guyo)
recon = np.clip(recon, 0, 1)
plt.imshow(np.squeeze(recon))
plt.grid(False)

plt.subplot(2, 1, 2)
z = tf.convert_to_tensor(tf.random.normal((1, latent_dim), mean=0, stddev=6))
recon = fRec.decoder.predict(z)
recon = np.clip(recon, 0, 1)
plt.imshow(np.squeeze(recon))
plt.grid(False)

plt.show()
# '''
