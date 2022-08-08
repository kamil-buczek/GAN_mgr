# Local
# External
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dropout, Dense, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Std
import os
import shutil
from datetime import date
from time import time, strftime


# Local


class GanNet(object):

    def __init__(self,
                 net_name: str,
                 batch_size: int,
                 image_width: int,
                 image_height: int,
                 number_of_channels: int,
                 latent_dimension: int,
                 training_data,
                 batches_per_epoch: int = None,
                 ):

        self._net_name: str = net_name
        self._batch_size: int = batch_size
        self._half_batch_size = int(self._batch_size / 2)

        if batches_per_epoch:
            self._batches_per_epoch: int = batches_per_epoch
        else:
            self._batches_per_epoch: int = int(training_data.shape[0] / self._batch_size)

        self._data_path = f'{os.getcwd()}/{self._net_name}_data'

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

        self._epoch_number_path = f'{self._data_path}/.epoch'
        self._epoch_number = 1

    def define_discriminator(self):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=self._input_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))

        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self._discriminator = model

    def define_generator(self):
        model = Sequential()
        init_width = self._generator_initial_image_width
        init_height = self._generator_initial_image_height

        n_nodes = 128 * init_width * init_height
        model.add(Dense(n_nodes, input_dim=self._latent_dimension))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((init_width, init_height, 128)))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

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

    def generate_real_images(self):
        # choose random instances
        images_indexes = np.random.randint(0, self._training_data_size, self._half_batch_size)
        real_image_from_training_data = self._training_data[images_indexes]
        # generate class labels
        expected_responses_for_real_images = np.ones((self._half_batch_size, 1))
        return real_image_from_training_data, expected_responses_for_real_images

    def generate_generator_inputs(self, size: int):
        # generate points in the latent space
        x_input = np.random.randn(self._latent_dimension * size)
        # reshape into a batch of inputs for the network
        generator_input = x_input.reshape(size, self._latent_dimension)
        return generator_input

    def generate_fake_images(self, size: int = None):

        if not size:
            size = self._half_batch_size

        # generate points in latent space
        generator_inputs = self.generate_generator_inputs(size=size)
        # predict outputs
        fake_images = self._generator.predict(generator_inputs)
        # create class labels
        expected_responses_for_fake_images = np.zeros((self._half_batch_size, 1))
        return fake_images, expected_responses_for_fake_images

    def save_sample_of_images(self):
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
        fig.set_facecolor('white')
        fig.savefig(os.path.join(os.getcwd(), f"{self._data_path}/sample_images/sample_{self._epoch_number}.png"))
        plt.close()

    def save_epoch_number_to_file(self):
        file_path = self._epoch_number_path
        with open(file_path, 'w') as file:
            file.write(f'{self._epoch_number}')

    def save_loss_data_to_files(self, disc_real, disc_fake, gen):

        mapping = {
            f'{self._data_path}/.loss_disc_fake': round(disc_fake, 2),
            f'{self._data_path}/.loss_disc_real': round(disc_real, 2),
            f'{self._data_path}/.loss_generator': round(gen, 2)
        }

        for file_name, data in mapping.items():
            with open(file_name, 'a') as file:
                file.write(f'{str(data)}\n')

    def save_time_to_file(self, start_time, end_time):

        seconds_elapsed = round(end_time - start_time, 2)
        hours, rest = divmod(seconds_elapsed, 3600)
        minutes, seconds = divmod(rest, 60)
        print(f'----> Epoch training time: {hours}h {minutes}m, {int(seconds)}s')

        time_file_path = f'{self._data_path}/.times'

        with open(time_file_path, 'a') as file:
            file.write(f'{self._epoch_number}={str(seconds_elapsed)}\n')

    def save_weights_to_files(self):

        epoch_num = self._epoch_number

        # Save once per 50 epoch
        if epoch_num % 50 == 0:
            self._generator.save_weights(f'{self._data_path}/weights/generator/weights_epoch_{epoch_num}.h5')
            self._discriminator.save_weights(
                f'{self._data_path}/weights/discriminator/weights_epoch_{epoch_num}.h5')
            self._gan.save_weights(f'{self._data_path}/weights/gan/weights_epoch_{epoch_num}.h5')

        # Save always
        self._generator.save_weights(f'{self._data_path}/weights/generator/weights_epoch_latest.h5')
        self._discriminator.save_weights(f'{self._data_path}/weights/discriminator/weights_epoch_latest.h5')
        self._gan.save_weights(f'{self._data_path}/weights/gan/weights_epoch_latest.h5')

    def save_models(self):
        self._generator.save(f'{self._data_path}/models/generator_flowers.h5')
        self._discriminator.save(f'{self._data_path}/models/discriminator_flowers.h5')
        self._gan.save(f'{self._data_path}/models/gan_flowers.h5')

    def load_loss_data_from_files(self) -> (list, list, list):

        disc_fake: list = []
        disc_real: list = []
        gen: list = []

        mapping = {
            f'{self._data_path}/.loss_disc_fake': disc_fake,
            f'{self._data_path}/.loss_disc_real': disc_real,
            f'{self._data_path}/.loss_generator': gen
        }

        for file_name, data in mapping.items():
            with open(file_name, 'r') as file:
                for line in file.readlines():
                    data.append(float(line.strip()))

        return mapping[f'{self._data_path}/.loss_disc_fake'], \
               mapping[f'{self._data_path}/.loss_disc_real'], \
               mapping[f'{self._data_path}/.loss_generator']

    # Train the generator and discriminator
    def train(self, number_of_epochs: int, load_past_model: bool = True):
        print(f'Number of images in dataset: {self._training_data_size}')
        print(f'Batches per epoch: {self._batches_per_epoch}')
        print(f'Half batch size is: {self._half_batch_size}')

        if load_past_model:
            self.load_model()
        else:
            self.archive_old_model()
            self.clear_files_structure()
            self.create_files_structure()

        for epoch_number in range(number_of_epochs):

            progbar = tf.keras.utils.Progbar(target=self._batches_per_epoch)

            print(f"\n----> Epoch: {self._epoch_number} {epoch_number + 1}/{number_of_epochs}")
            start_time = time()
            print(f'---> Start time is: {strftime("%H:%M:%S")}')

            discriminator_real_images_loss = 0
            discriminator_fake_images_loss = 0
            generator_loss = 0
            discriminator_real_images_accuracy = 0
            discriminator_fake_images_accuracy = 0

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
                generator_random_input_vector = self.generate_generator_inputs(self._batch_size)
                # create inverted labels for the fake samples
                generator_expected_answers = np.ones((self._batch_size, 1))
                # update the generator via the discriminator's error
                generator_loss = self._gan.train_on_batch(generator_random_input_vector, generator_expected_answers)
                # summarize loss on this batch
                progbar.update(batch_number + 1)

                # Save loss function data after batch
                self.save_loss_data_to_files(discriminator_real_images_loss, discriminator_fake_images_loss,
                                             generator_loss)

            end_time = time()
            print(f'---> End time is: {strftime("%H:%M:%S")}')

            # Results after epoch
            print(f"\nD_real_loss: {discriminator_real_images_loss}"
                  f" D_fake_loss: {discriminator_fake_images_loss}"
                  f" G_loss: {generator_loss}")

            print(f"D_real_acc: {discriminator_real_images_accuracy}"
                  f" D_fake_acc: {discriminator_fake_images_accuracy}")

            self.save_time_to_file(start_time, end_time)
            self.save_weights_to_files()
            self.save_sample_of_images()
            self.save_epoch_number_to_file()
            self._epoch_number = self._epoch_number + 1
        # After full training
        self.save_models()
        self.plot_loss()
        self.get_training_time()

    def plot_loss(self):

        disc_fake_data, disc_real_data, generator_data = self.load_loss_data_from_files()

        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plt.plot([_ for _ in disc_real_data], color='blue', linewidth=0.5)
        plt.plot([_ for _ in disc_fake_data], color='green', linewidth=0.5)
        plt.plot([_ for _ in generator_data], color='red', linewidth=0.5)

        max_value = max([max(disc_real_data), max(disc_fake_data), max(generator_data)])

        plt.legend(('Discriminator real loss', 'Discriminator fake loss', 'Generator loss'),
                   fontsize=15)
        plt.title('Loss function', fontsize=15)

        plt.xlabel('batch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlim(0, len(disc_real_data))
        plt.ylim(0, max_value + 1)

        fig.savefig(os.path.join(os.getcwd(), f"{self._data_path}/plots/loss_{self._epoch_number}.png"))

    def load_epoch_number_from_file(self) -> int:
        file_path = self._epoch_number_path
        with open(file_path) as file:
            epoch = file.read()
        epoch_number = int(epoch.strip())
        print(f'----> Load epoch number: {epoch_number} from file {file_path}')
        return epoch_number + 1

    def load_weights(self):
        self._gan.load_weights(f'{self._data_path}/weights/gan/weights_epoch_latest.h5')
        self._generator.load_weights(f'{self._data_path}/weights/generator/weights_epoch_latest.h5')
        self._discriminator.load_weights(f'{self._data_path}/weights/discriminator/weights_epoch_latest.h5')

    def load_model(self) -> None:
        # Load epoch number from file
        self._epoch_number = self.load_epoch_number_from_file()
        # Load weights
        self.load_weights()

    def create_files_structure(self):

        directories = [
            'sample_images',
            'plots',
            'models',
            'weights/gan',
            'weights/generator',
            'weights/discriminator',
            'viz'
        ]

        for _ in directories:
            os.makedirs(f'{self._data_path}/{_}', exist_ok=True)

    def clear_files_structure(self):
        shutil.rmtree(self._data_path, ignore_errors=True)

    def archive_old_model(self):
        if os.path.exists(self._data_path):
            print(f'---> Archive old networks data: {self._data_path}')

            archive_name = f'{self._net_name}_{date.today()}'
            shutil.make_archive(archive_name, 'zip', self._data_path)
            shutil.move(f'{archive_name}.zip',  f'{os.getcwd()}/old_networks/{archive_name}.zip' )

    def visualize_models(self):
        tf.keras.utils.plot_model(self._gan, to_file=f'{self._data_path}/viz/gan.png', show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self._discriminator, to_file=f'{self._data_path}/viz/discriminator.png',
                                  show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self._generator, to_file=f'{self._data_path}/viz/generator.png',
                                  show_shapes=True,
                                  show_layer_names=True)

    def get_training_time(self):

        time_file_path = f'{self._data_path}/.times'
        sum_in_sec = 0

        with open(time_file_path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split('=')
                seconds = float(line[1])
                sum_in_sec += seconds

        hours, rest = divmod(sum_in_sec, 3600)
        minutes, seconds = divmod(rest, 60)
        print(f'----> Total training time: {hours}h {minutes}m, {int(seconds)}s')