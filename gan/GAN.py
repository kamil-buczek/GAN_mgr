# External
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dropout, Dense, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Std
import os


# Local


class GanNet(object):

    def __init__(self,
                 batch_size: int,
                 batches_per_epoch: int,
                 image_width: int,
                 image_height: int,
                 number_of_channels: int,
                 latent_dimension: int,
                 training_data,
                 epoch_number_file: str = '.epoch',
                 data_path: str = f'{os.getcwd()}/data'
                 ):

        self._batch_size: int = batch_size
        self._batches_per_epoch: int = batches_per_epoch
        self._data_path = data_path

        self._image_width: int = image_width
        self._image_height: int = image_height
        self._number_of_channels: int = number_of_channels
        self._latent_dimension: int = latent_dimension

        self._input_shape: tuple = (self._image_width, self._image_height, self._number_of_channels)
        self._generator_initial_image_width = int(self._image_width / 4)
        self._generator_initial_image_height = int(self._image_height / 4)

        self._discriminator = None
        self._generator = None
        self._gan = None

        self._training_data = training_data
        self._training_data_size = len(self._training_data)

        self._epoch_number_file = epoch_number_file

        self._discriminator_real_loss_list = []
        self._discriminator_fake_loss_list = []
        self._generator_loss_list = []

    def define_discriminator(self):
        model = Sequential()

        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=self._input_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self._discriminator = model

    def define_generator(self):
        model = Sequential()
        # foundation for init_width x init_height image
        init_width = self._generator_initial_image_width
        init_height = self._generator_initial_image_height

        n_nodes = 128 * init_width * init_height
        model.add(Dense(n_nodes, input_dim=self._latent_dimension))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((init_width, init_height, 128)))
        # upsample to 2*init_width x 2*init_height
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 2*2*init_width x 2*2*init_height
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # generate
        # TODO nie wiem czy tu trzeba zmienic to 7 7 na  init_width i init_height
        model.add(Conv2D(3, (7, 7), activation='tanh', padding='same'))
        self._generator = model

    def define_gan(self):
        # make weights in the discriminator not trainable
        self._discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self._generator)
        # add the discriminator
        model.add(self._discriminator)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        self._gan = model

    # select real samples
    def generate_real_images(self):
        # choose random instances
        # ix = np.random.randint(0, self._training_data_size, n_samples)
        images_indexes = np.random.randint(0, self._training_data_size, self._batch_size)
        real_image_from_training_data = self._training_data[images_indexes]
        # generate class labels
        # y = np.ones((n_samples, 1))
        expected_responses_for_real_images = np.ones((self._batch_size, 1))
        return real_image_from_training_data, expected_responses_for_real_images

    def generate_generator_inputs(self):
        """
        Generate points in latent space as input for the generator

        :return:
        """
        # generate points in the latent space
        x_input = np.random.randn(self._latent_dimension * self._batch_size)
        # reshape into a batch of inputs for the network
        generator_input = x_input.reshape(self._batch_size, self._latent_dimension)
        return generator_input

    def generate_fake_images(self):
        """
        Use the generator to generate n fake examples, with class labels

        :return:
        """
        # generate points in latent space
        generator_inputs = self.generate_generator_inputs()
        # predict outputs
        fake_images = self._generator.predict(generator_inputs)
        # create class labels
        exoected_responses_for_fake_images = np.zeros((self._batch_size, 1))
        return fake_images, exoected_responses_for_fake_images

    def save_sample_of_images(self, epoch_number):
        rows = 5
        cols = 5
        noise = np.random.normal(0, 1, (rows * cols, self._latent_dimension))
        gen_imgs = self._generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        cnt = 0

        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(os.getcwd(), f"{self._data_path}/sample_images/sample_{epoch_number}.png"))
        plt.close()

    def save_epoch_number_to_file(self, epoch_number):
        file_name = self._epoch_number_file
        with open(file_name, 'w') as file:
            file.write(f'{epoch_number}')
        # print(f'----> Save epoch number {epoch_number} to file {file_name}')

    def load_epoch_number_from_file(self) -> int:
        file_name = self._epoch_number_file
        with open(file_name) as file:
            epoch = file.read()
        epoch_number = int(epoch.strip())
        print(f'----> Load epoch number: {epoch_number} from file {file_name}')
        return epoch_number

    # train the generator and discriminator
    def train(self, number_of_epochs: int, load_past_model: bool = True):
        print("Dataset size:", self._training_data_size)
        print(f'Batches per epoch: {self._batches_per_epoch}')

        if load_past_model:
            past_epochs_number = self.load_epoch_number_from_file() + 1
            self.load_weights()
        else:
            past_epochs_number = 0

        for epoch_number in range(number_of_epochs):

            progbar = tf.keras.utils.Progbar(target=self._batches_per_epoch)
            actual_epoch_numer = epoch_number + past_epochs_number

            print(f"----> Epoch: {actual_epoch_numer}")

            epoch_discriminator_real_images_loss = 0
            epoch_discriminator_fake_images_loss = 0
            epoch_generator_loss = 0
            epoch_discriminator_real_images_accuracy = 0
            epoch_discriminator_fake_images_accuracy = 0
            # epoch_generator_accuracy = 0

            for batch_number in range(self._batches_per_epoch):
                # get randomly selected 'real' samples
                real_images, real_answers = self.generate_real_images()
                # update discriminator model weights
                discriminator_real_images_loss, discriminator_real_images_accuracy = \
                    self._discriminator.train_on_batch(real_images, real_answers)
                # generate 'fake' examples
                fake_images, fake_answers = self.generate_fake_images()
                # update discriminator model weights
                discriminator_fake_images_loss, discriminator_fake_images_accuracy = \
                    self._discriminator.train_on_batch(fake_images, fake_answers)
                # prepare points in latent space as input for the generator
                generator_random_input_vector = self.generate_generator_inputs()
                # create inverted labels for the fake samples
                generator_expected_answers = np.ones((self._batch_size, 1))
                # update the generator via the discriminator's error
                generator_loss = self._gan.train_on_batch(generator_random_input_vector, generator_expected_answers)
                # summarize loss on this batch
                progbar.update(batch_number+1)

                self._discriminator_real_loss_list.append(discriminator_real_images_loss)
                self._discriminator_fake_loss_list.append(discriminator_fake_images_loss)
                self._generator_loss_list.append(generator_loss)

                epoch_discriminator_real_images_loss = discriminator_real_images_loss
                epoch_discriminator_fake_images_loss = discriminator_fake_images_loss
                epoch_generator_loss = generator_loss
                epoch_discriminator_real_images_accuracy = discriminator_real_images_accuracy
                epoch_discriminator_fake_images_accuracy = discriminator_fake_images_accuracy
                # epoch_generator_accuracy = generator_accuracy

            print(f"\nD_real_loss: {epoch_discriminator_real_images_loss}"
                  f" D_fake_loss: {epoch_discriminator_fake_images_loss}"
                  f" G_loss: {epoch_generator_loss}")

            print(f"D_real_acc: {epoch_discriminator_real_images_accuracy}"
                  f" D_fake_acc: {epoch_discriminator_fake_images_accuracy}")
                  #f" G_acc: {epoch_generator_accuracy}")

            self.save_sample_of_images(epoch_number=actual_epoch_numer)
            self._generator.save_weights(f'{self._data_path}/weights/generator/weights_epoch_{actual_epoch_numer}.h5')
            self._generator.save_weights(f'{self._data_path}/weights/generator/weights_epoch_latest.h5')
            self._discriminator.save_weights(f'{self._data_path}/weights/discriminator/weights_epoch_{actual_epoch_numer}.h5')
            self._discriminator.save_weights(f'{self._data_path}/weights/discriminator/weights_epoch_latest.h5')
            self._gan.save_weights(f'{self._data_path}/weights/gan/weights_epoch_{actual_epoch_numer}.h5')
            self._gan.save_weights(f'{self._data_path}/weights/gan/weights_epoch_latest.h5')
            self.save_epoch_number_to_file(actual_epoch_numer)
        # save the generator model
        self._generator.save(f'{self._data_path}/models/generator_flowers.h5')
        self._discriminator.save(f'{self._data_path}/models/discriminator_flowers.h5')
        self._gan.save(f'{self._data_path}/models/gan_flowers.h5')

    def load_weights(self):
        self._gan.load_weights(f'{self._data_path}/weights/gan/weights_epoch_latest.h5')
        self._generator.load_weights(f'{self._data_path}/weights/generator/weights_epoch_latest.h5')
        self._discriminator.load_weights(f'{self._data_path}/weights/discriminator/weights_epoch_latest.h5')

    def visualize_model(self):
        tf.keras.utils.plot_model(self._gan, to_file=f'{self._data_path}/viz/gan.png', show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self._discriminator, to_file=f'{self._data_path}/viz/discriminator.png',
                                  show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self._generator, to_file=f'{self._data_path}/viz/generator.png',
                                  show_shapes=True,
                                  show_layer_names=True)

    def plot_loss(self):

        epoch_number = self.load_epoch_number_from_file()

        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plt.plot([_ for _ in self._discriminator_real_loss_list], color='blue', linewidth=0.5)
        plt.plot([_ for _ in self._discriminator_fake_loss_list], color='green', linewidth=0.5)
        plt.plot([_ for _ in self._generator_loss_list], color='red', linewidth=0.5)

        plt.legend(('Discriminator real loss', 'Discriminator fake loss', 'Generator loss'),
                   fontsize=15)
        plt.title('Loss function', fontsize=15)

        plt.xlabel('batch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlim(0, len(self._discriminator_real_loss_list))
        plt.ylim(0, 3)

        fig.savefig(os.path.join(os.getcwd(), f"{self._data_path}/plots/loss_{epoch_number}.png"))

    def create_files_structure(self):


        directories = [
            'sample_images',
            'plots',
            'models',
            'weights/gan',
            'weights/generator',
            'weights/discriminator'
        ]

        for _ in directories:
            os.makedirs(f'{self._data_path}/{_}', exist_ok=True)

    def clear_files_structure(self):
        shutil.rmtree(self._data_path)