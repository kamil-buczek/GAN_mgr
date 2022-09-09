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
                 learning_rate_disc: float,
                 learning_rate_gan: float,
                 dropout_rate: float,
                 num_conv_layers: int,
                 number_of_channels: int,
                 latent_dimension: int,
                 training_data,
                 batches_per_epoch: int = None,
                 ):

        self._net_name: str = net_name
        self._batch_size: int = batch_size
        self._half_batch_size = int(self._batch_size / 2)
        self._dropout_rate = dropout_rate
        self._learning_rate_disc = learning_rate_disc
        self._learning_rate_gan = learning_rate_gan

        if batches_per_epoch:
            self._batches_per_epoch: int = batches_per_epoch
        else:
            self._batches_per_epoch: int = int(training_data.shape[0] / self._batch_size)

        self._data_path = f'{os.getcwd()}/{self._net_name}_data'

        self._num_conv_layers = num_conv_layers

        self._image_width: int = image_width
        self._image_height: int = image_height
        self._number_of_channels: int = number_of_channels
        self._latent_dimension: int = latent_dimension

        self._input_shape: tuple = (self._image_width, self._image_height, self._number_of_channels)
        self._start_image_width = int(self._image_width / (2**self._num_conv_layers))
        self._start_image_height = int(self._image_height / (2**self._num_conv_layers))

        self._discriminator = None
        self._generator = None
        self._gan = None

        self._training_data = training_data
        self._training_data_size = len(self._training_data)

        self._epoch_number_path = f'{self._data_path}/.epoch'
        self._epoch_number = 1

    def define_discriminator(self):
        pass

    def define_generator(self):
        pass

    def define_gan(self):
        pass

    def generate_real_images(self):
        pass

    def generate_generator_inputs(self, size: int):
        pass

    def generate_fake_images(self, size: int = None):
        pass

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

    def save_accuracy_data_to_files(self, disc_real, disc_fake):

        mapping = {
            f'{self._data_path}/.acc_disc_fake': round(disc_fake, 2),
            f'{self._data_path}/.acc_disc_real': round(disc_real, 2),
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

        epoch_number = self._epoch_number

        # Save once per 50 epoch
        if epoch_number % 50 == 0:
            self._generator.save_weights(f'{self._data_path}/weights/generator/weights_epoch_{epoch_number}.h5')
            self._discriminator.save_weights(
                f'{self._data_path}/weights/discriminator/weights_epoch_{epoch_number}.h5')
            self._gan.save_weights(f'{self._data_path}/weights/gan/weights_epoch_{epoch_number}.h5')

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
        pass

    def plot_loss(self):

        # Wczytanie danych z plików
        disc_fake_data, disc_real_data, generator_data = self.load_loss_data_from_files()

        # Tworzenie wykresu
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

        # Zapisanie wykresu
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

    def create_directories(self):

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