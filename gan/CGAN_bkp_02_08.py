from gan.GAN import GanNet
import os
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, \
    Concatenate, Conv2D, LeakyReLU, Flatten, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from matplotlib import pyplot


class CGanNet(GanNet):

    def __init__(self,
                 batch_size: int,
                 batches_per_epoch: int,
                 image_width: int,
                 image_height: int,
                 number_of_channels: int,
                 latent_dimension: int,
                 training_data,
                 labels_data,
                 labels_names,
                 number_of_classes: int,
                 epoch_number_file: str = '.epoch',
                 data_path: str = f'{os.getcwd()}/cgan_data',
                 ):

        super(CGanNet, self).__init__(batch_size,
                                     batches_per_epoch,
                                     image_width,
                                     image_height,
                                     number_of_channels,
                                     latent_dimension,
                                     training_data,
                                     epoch_number_file,
                                     data_path)

        self._number_of_classes = number_of_classes
        self._labels_data = labels_data
        self._labels_names = labels_names

    def define_discriminator(self):

        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self._number_of_classes, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = self._input_shape[0] * self._input_shape[1]
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((self._input_shape[0], self._input_shape[1], 1))(li)
        # image input
        in_image = Input(shape=self._input_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        # downsample
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self._discriminator = model

    def define_generator(self):

        init_width = self._generator_initial_image_width
        init_height = self._generator_initial_image_height

        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self._number_of_classes, 50)(in_label)
        # linear multiplication
        n_nodes = init_width * init_height
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((init_width, init_height, 1))(li)
        # image generator input
        in_lat = Input(shape=(self._latent_dimension,))
        # foundation for 7x7 image
        n_nodes = 128 * init_width * init_height
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((init_width, init_height, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x14
        gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # output
        out_layer = Conv2D(self._number_of_channels, (7, 7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        self._generator = model

    def define_gan(self):

        # make weights in the discriminator not trainable
        self._discriminator.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = self._generator.input
        # get image output from the generator model
        gen_output = self._generator.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = self._discriminator([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self._gan = model

    def generate_real_images(self):
        # choose random instances
        indexes = np.random.randint(0, self._training_data_size, self._batch_size)
        real_image_from_training_data = self._training_data[indexes]
        real_label_from_labels_data = self._labels_data[indexes]

        # generate class labels
        expected_responses_for_real_images = np.ones((self._batch_size, 1))
        return [real_image_from_training_data, real_label_from_labels_data], expected_responses_for_real_images

    def generate_generator_inputs(self, size=None):

        if not size:
            size = self._batch_size

        # generate points in the latent space
        x_input = np.random.randn(self._latent_dimension * size)
        # reshape into a batch of inputs for the network
        generator_input = x_input.reshape(size, self._latent_dimension)

        # generate labels
        labels = np.random.randint(0, self._number_of_classes, size)
        return [generator_input, labels]

    def generate_fake_images(self):
        # generate points in latent space
        generator_inputs, labels_input = self.generate_generator_inputs()
        # predict outputs
        fake_images = self._generator.predict([generator_inputs, labels_input])
        # create class labels
        expected_responses_for_fake_images = np.zeros((self._batch_size, 1))
        return [fake_images, labels_input], expected_responses_for_fake_images

    def load_model(self) -> int:
        epoch_num = self.load_epoch_number_from_file() + 1
        self.load_weights()
        return epoch_num

    def train(self, number_of_epochs: int, load_past_model: bool = True):
        print("Dataset size:", self._training_data_size)
        print(f'Batches per epoch: {self._batches_per_epoch}')

        if load_past_model:
            past_epochs_number = self.load_model()
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
                [real_images, real_labels], real_answers = self.generate_real_images()
                # update discriminator model weights
                discriminator_real_images_loss, discriminator_real_images_accuracy = \
                    self._discriminator.train_on_batch([real_images, real_labels], real_answers)
                # generate 'fake' examples
                [fake_images, fake_labels], fake_answers = self.generate_fake_images()
                # update discriminator model weights
                discriminator_fake_images_loss, discriminator_fake_images_accuracy = \
                    self._discriminator.train_on_batch([fake_images, fake_labels], fake_answers)
                # prepare points in latent space as input for the generator
                [generator_random_input_vector, labels_input] = self.generate_generator_inputs()
                # create inverted labels for the fake samples
                generator_expected_answers = np.ones((self._batch_size, 1))
                # update the generator via the discriminator's error
                generator_loss = self._gan.train_on_batch([generator_random_input_vector, labels_input],
                                                          generator_expected_answers)
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

    def save_sample_of_images(self, epoch_number):
        rows = 5
        cols = 5

        generator_inputs, labels = self.generate_generator_inputs(size=25)
        random_labels = self.get_random_labels_list(5)
        labels = np.asarray(random_labels)
        gen_imgs = self._generator.predict([generator_inputs, labels])

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        cnt = 0

        for i in range(rows):
            for j in range(cols):
                axs[i, j].set_title(f'({labels[cnt]}) {self._labels_names[str(labels[cnt])]}')
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.set_facecolor('white')
        fig.savefig(os.path.join(os.getcwd(), f"{self._data_path}/sample_images/sample_{epoch_number}.png"))
        plt.close()

    def get_random_labels_list(self, size: int = 5) -> list:

        random_labels_part = []

        while len(random_labels_part) < size:
            random_label = random.randint(0, self._number_of_classes - 1)
            if random_label not in random_labels_part:
                random_labels_part.append(random_label)

        random_labels_groups = []

        for _ in range(5):
            random_labels_groups.extend(random_labels_part)

        return random_labels_groups

    def show_sample_images(self):
        x_fake, _ = self.generate_fake_images()
        images = (x_fake[0] + 1) / 2.0
        # gen_imgs = np.clip(images, 0, 1)

        rows = 5
        cols = 5
        cnt = 0

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        for i in range(rows):
            for j in range(cols):
                axs[i, j].set_title(f'({x_fake[1][cnt]}) {self._labels_names[str(x_fake[1][cnt])]}')
                axs[i, j].imshow(images[cnt], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.set_facecolor('white')
        plt.show()

    def show_sample_image_one(self, label_num: int):
        noise, _ = self.generate_generator_inputs(size=1)
        label_arr = np.array([label_num])
        image = self._generator.predict([noise, label_arr])
        # gen_image = np.clip(image, 0, 1)
        image = (image + 1) / 2.0
        plt.axis('off')
        plt.imshow(np.squeeze(image), cmap='gray')
