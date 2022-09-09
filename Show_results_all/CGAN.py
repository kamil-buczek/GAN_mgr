# Std
import os
import random
from time import time, strftime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# External
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, \
    Concatenate, Conv2D, LeakyReLU, Flatten, Dropout, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Local
from GAN import GanNet


class CGanNet(GanNet):

    def __init__(self,
                 net_name: str,
                 batch_size: int,
                 image_width: int,
                 image_height: int,
                 learning_rate_disc: float,
                 learning_rate_gan: float,
                 dropout_rate: float,
                 generator_dense_units: int,
                 num_conv_layers: int,
                 batch_norm: bool,
                 number_of_channels: int,
                 latent_dimension: int,
                 training_data,
                 labels_data: dict,
                 labels_names,
                 number_of_classes: int,
                 batches_per_epoch: int = None,
                 kernel_size: int = 7
                 ):

        super(CGanNet, self).__init__(net_name,
                                      batch_size,
                                      image_width,
                                      image_height,
                                      learning_rate_disc,
                                      learning_rate_gan,
                                      dropout_rate,
                                      num_conv_layers,
                                      number_of_channels,
                                      latent_dimension,
                                      training_data,
                                      batches_per_epoch,
                                      )

        self._number_of_classes = number_of_classes
        self._labels_data = labels_data
        self._labels_names = labels_names
        self._generator_dense_units = generator_dense_units
        self._batch_norm = batch_norm
        self._kernel_size = kernel_size

    def define_discriminator(self):
        # Przekształcenie etykiety do postaci kanału obrazu
        label_input = Input(shape=(1,), name='Disc-Label-Input-Layer')
        label_layer = Embedding(self._number_of_classes, 50, name='Disc-Label-Embedding-Layer')(label_input)
        n_nodes = self._input_shape[0] * self._input_shape[1]
        label_layer = Dense(n_nodes, name='Disc-Label-Dense_layer')(label_layer)
        label_layer = Reshape((self._input_shape[0], self._input_shape[1], 1), name='Disc-Label-Reshape-Layer')(label_layer)

        # Obraz do oceny przez dyskryminator
        image_input = Input(shape=self._input_shape, name='Disc-Image-Input-Layer')

        # Połączenie obrazu z etykietą
        image_layer = Concatenate(name='Disc-Combine-Layers')([image_input, label_layer])

        for _ in range(self._num_conv_layers):
            image_layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        name=f'Disc-Downsample-{_ + 1}-Layer')(image_layer)
            image_layer = LeakyReLU(alpha=0.2, name=f'Disc-Downsample-{_ + 1}-Layer-Activation')(image_layer)

        # Spłaszczenie do postaci pojedynczego wektora
        flatten_image = Flatten(name='Disc-Flatten-Layer')(image_layer)
        # Dodanie warstwy dropout
        if self._dropout_rate:
            flatten_image = Dropout(self._dropout_rate, name='Disc-Flatten-Layer-Dropout')(flatten_image)
        else:
            print("Nie ustawiono dropout_rate. Usuwam warstwe Dropout z modelu dyskryminatora")

        # Warstwa wyjsciowa jako liczba z zakresu [0,1]
        output_layer = Dense(1, activation='sigmoid', name='Disc-Output-Layer')(flatten_image)

        # Uworzenie modelu dyskryminatora
        discriminator_model = Model([image_input, label_input], output_layer, name='Discriminator-Model')

        # Kompilacja modelu dyskryminatora
        optymlizator = Adam(learning_rate=self._learning_rate_disc, beta_1=0.5)
        discriminator_model.compile(loss='binary_crossentropy', optimizer=optymlizator, metrics=['accuracy'])
        self._discriminator = discriminator_model

    def define_generator(self):
        # Początkowa rozdzielczość obrazu, przed przejściem przez warstwy konwolucyjne
        start_width = self._start_image_width
        start_height = self._start_image_height

        # Przekształcenie etykiety do postaci kanału obrazu
        label_input = Input(shape=(1,), name='Gen-Label-Input-layer')
        label_layer = Embedding(self._number_of_classes, 50, name='Gen-Label-Embedding-Layer')(label_input)
        n_nodes = start_width * start_height
        label_layer = Dense(n_nodes, name='Gen-Label-Dense-Layer')(label_layer)
        label_layer = Reshape((start_width, start_height, 1), name='Gen-Label-Reshape_Layer')(label_layer)

        # Przekształcenie latent wektora do postaci obrazu
        latent_vector_input = Input(shape=(self._latent_dimension,), name='Gen-Latent-Input-Layer')
        n_nodes = self._generator_dense_units * start_width * start_height
        image_layer = Dense(n_nodes, name='Gen-Foundation-Layer')(latent_vector_input)
        if self._batch_norm:
            image_layer = BatchNormalization(momentum=0.9)(image_layer)
        image_layer = LeakyReLU(alpha=0.2, name='Gen-Foundation-Layer-Activation')(image_layer)
        image_layer = Reshape((start_width, start_height, self._generator_dense_units), name='Gen-Foundation-Layer-Reshape')(image_layer)

        # Połączenie obrazu z etykietą
        image_layer = Concatenate(name='Gen-Combine-Layer')([image_layer, label_layer])

        for _ in range(self._num_conv_layers):
            image_layer = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                  name=f'Gen-Upsample-{_ + 1}-Layer')(image_layer)
            if self._batch_norm:
                image_layer = BatchNormalization(momentum=0.9, name=f'Gen-Upsample--{_ + 1}-Layer-Batch-Normalization')(image_layer)
            image_layer = LeakyReLU(alpha=0.2, name=f'Gen-Upsample-{_ + 1}-Layer-Activation')(image_layer)

        # Wygenerowany obraz na wyjściu z sieci
        output_layer = Conv2D(filters=self._number_of_channels, kernel_size=(self._kernel_size, self._kernel_size), activation='tanh', padding='same',
                           name='Gen-Output-Layer')(image_layer)

        # Utworzenie modelu generatora
        generator_model = Model([latent_vector_input, label_input], output_layer, name='Generator-Model')
        self._generator = generator_model

    def define_gan(self):

        # Podczas treningu połączonego modelu ma trenować tylko generator, więc blokujemy uczenie dla dyskryminatora
        self._discriminator.trainable = False

        # Szumy i etykiety z wejścia generatora
        generator_input_noise, generator_input_label = self._generator.input

        # Obraz wyjściowy z generatora
        generator_output = self._generator.output

        # Connect image output and label input from generator as inputs to discriminator
        gan_output = self._discriminator([generator_output, generator_input_label])

        # Połączony model na wejściu otrzymuję losowy szum i etykietę i zwraca liczbę z zakresu [0,1]
        model_gan = Model([generator_input_noise, generator_input_label], gan_output, name='Conditional-DCGAN')

        # Kompilacja modelu
        optymalizator = Adam(learning_rate=self._learning_rate_gan, beta_1=0.5)
        model_gan.compile(loss='binary_crossentropy', optimizer=optymalizator)
        self._gan = model_gan

    def generate_real_images(self):
        # Wylosuj obrazy i ich etykiety ze zbioru treningowego
        random_numbers = np.random.randint(0, self._training_data_size, self._half_batch_size)
        random_images = self._training_data[random_numbers]
        random_labels = self._labels_data[random_numbers]

        # Oczekiwana wartosc zwrocona prze dyskryminatora dla prawdziwego obrazu to 1
        expected_responses = np.ones((self._half_batch_size, 1))
        return [random_images, random_labels], expected_responses

    def generate_generator_inputs(self, size: int):
        # Wylosuj liczby z rozkładu normalnego i przekształć do postaci wektora
        random_latent_numbers = np.random.randn(self._latent_dimension * size)
        generator_input = random_latent_numbers.reshape(size, self._latent_dimension)
        # Losowanie etykiety
        labels_input = np.random.randint(0, self._number_of_classes, size)
        return [generator_input, labels_input]

    def generate_fake_images(self, size: int = None):

        if not size:
            size = self._half_batch_size

        # Latent wektor i etykieta
        latent_inputs, labels_input = self.generate_generator_inputs(size=size)
        # Wygeneruj obraz
        generated_image = self._generator.predict([latent_inputs, labels_input])
        # Dla fałszywych obrazow dyskryminator ma zwrócić 0
        expected_response = np.zeros((self._half_batch_size, 1))
        return [generated_image, labels_input], expected_response

    def train(self, number_of_epochs: int, load_past_model: bool = True):
        print(f'Number of images in dataset: {self._training_data_size}')
        print(f'Batches per epoch: {self._batches_per_epoch}')
        print(f'Half batch size is: {self._half_batch_size}')

        if load_past_model:
            self.load_model()
        else:
            self.archive_old_model()
            self.clear_files_structure()
            self.create_directories()

        for epoch_number in range(number_of_epochs):
            print('------------------------------------------------------------')
            progbar = tf.keras.utils.Progbar(target=self._batches_per_epoch)
            print(f"---> Epoch: {self._epoch_number} {epoch_number + 1}/{number_of_epochs}")
            start_time = time()
            print(f'---> Start time is: {strftime("%H:%M:%S")}')

            discriminator_real_images_loss = 0
            discriminator_fake_images_loss = 0
            generator_loss = 0
            discriminator_real_images_accuracy = 0
            discriminator_fake_images_accuracy = 0

            for batch_number in range(self._batches_per_epoch):
                # Uczenie na prawdziwych obrazach
                [real_images, real_labels], real_answers = self.generate_real_images()
                discriminator_real_images_loss, discriminator_real_images_accuracy = \
                    self._discriminator.train_on_batch([real_images, real_labels], real_answers)

                # Uczenie na fałszywych obrazach
                [fake_images, fake_labels], fake_answers = self.generate_fake_images()
                discriminator_fake_images_loss, discriminator_fake_images_accuracy = \
                    self._discriminator.train_on_batch([fake_images, fake_labels], fake_answers)

                # Wejście dla generatora
                [generator_random_input_vector, labels_input] = self.generate_generator_inputs(size=self._batch_size)
                # Chcemy oszukać dyskryminatora, czyli ma zwrócić 1
                generator_expected_answers = np.ones((self._batch_size, 1))
                # Uczenie generatora
                generator_loss = self._gan.train_on_batch([generator_random_input_vector, labels_input],
                                                          generator_expected_answers)
                # Update progress bar
                progbar.update(batch_number + 1)

                # Save loss data
                self.save_loss_data_to_files(discriminator_real_images_loss, discriminator_fake_images_loss,
                                             generator_loss)

                self.save_accuracy_data_to_files(discriminator_real_images_accuracy, discriminator_fake_images_accuracy)

            end_time = time()
            print(f'---> End time is: {strftime("%H:%M:%S")}')

            # Results after one epoch
            print(f"---> D_real_loss: {discriminator_real_images_loss}"
                  f" D_fake_loss: {discriminator_fake_images_loss}"
                  f" G_loss: {generator_loss}")

            print(f"----> D_real_acc: {discriminator_real_images_accuracy}"
                  f" D_fake_acc: {discriminator_fake_images_accuracy}")

            self.save_time_to_file(start_time, end_time)
            self.save_weights_to_files()
            self.save_sample_of_images_with_labels()
            self.save_epoch_number_to_file()
            self._epoch_number = self._epoch_number + 1
        # After full training
        self.save_models()
        self.plot_loss()
        self.get_training_time()

    def save_sample_of_images_with_labels(self):

        number_of_rows = 5
        number_of_columns = 5
        counter = 0

        generator_inputs, labels = \
            self.generate_generator_inputs(size=5*number_of_rows)
        random_labels = self.get_random_labels_list(5)

        labels = np.asarray(random_labels)
        images = self._generator.predict([generator_inputs, labels])

        images = (images + 1) / 2.0
        images = (images * 255).astype(np.uint8)

        figure, sub_plot = plt.subplots(number_of_rows, number_of_columns, figsize=(15, 15))
        for i in range(number_of_rows):
            for j in range(number_of_columns):
                sub_plot[i, j].set_title(f'({labels[counter]}) '
                                    f'{self._labels_names[labels[counter]]}')
                sub_plot[i, j].imshow(np.squeeze(images[counter, :, :, :]),
                                 cmap='gray')
                sub_plot[i, j].axis('off')
                counter += 1
        figure.set_facecolor('white')
        figure.savefig(os.path.join(os.getcwd(), f"{self._data_path}/sample_images/sample_{self._epoch_number}.png"))
        plt.close()

    def get_random_labels_list(self, size: int = 5) -> list:

        if size > self._number_of_classes:
            size = self._number_of_classes

        random_labels_part = []
        while len(random_labels_part) < size:
            random_label = random.randint(0, self._number_of_classes-1)
            if random_label not in random_labels_part:
                random_labels_part.append(random_label)

        if self._number_of_classes == 3:
            random_labels_part = [0, 1, 1, 2, 2]

        random_labels_groups = []
        for _ in range(5):
            random_labels_groups.extend(random_labels_part)
        return random_labels_groups

    def show_sample_images_with_labels(self, old_img_format: bool = True):

        rows = 5
        cols = 5
        cnt = 0

        generator_inputs, _ = self.generate_generator_inputs(size=25)
        random_labels = self.get_random_labels_list(5)
        labels = np.asarray(random_labels)
        gen_imgs = self._generator.predict([generator_inputs, labels])

        if old_img_format:
            gen_imgs = np.clip(gen_imgs, 0, 1)
        else:
            gen_imgs = (gen_imgs + 1) / 2.0
            gen_imgs = (gen_imgs*255).astype(np.uint8)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        for i in range(rows):
            for j in range(cols):
                axs[i, j].set_title(f'({labels[cnt]}) {self._labels_names[labels[cnt]]}')
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.set_facecolor('white')
        plt.show()

    def show_one_image_with_label(self, label_num: int, old_img_format: bool = True):
        noise, _ = self.generate_generator_inputs(size=1)
        label = np.asarray([label_num])
        image = self._generator.predict([noise, label])

        if old_img_format:
            image = np.clip(image, 0, 1)
        else:
            image = (image + 1) / 2.0
            image = (image * 255).astype(np.uint8)
        plt.axis('off')
        plt.imshow(np.squeeze(image), cmap='gray')
