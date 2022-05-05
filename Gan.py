import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

run_folder = 'run'


def get_activation(activation):
    if activation == 'leaky_relu':
        layer = tf.keras.layers.LeakyReLU(alpha=0.2)
    else:
        layer = tf.keras.layers.Activation(activation)
    return layer


def set_trainable(m, val):
    m.trainable = val
    for layer in m.layers:
        layer.trainable = val


def load_mnist_gan():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


def get_epoch_number() -> int:
    file_name = f'{run_folder}/epoch.txt'
    with open(file_name) as file:
        epoch = file.read()
    return int(epoch.strip())


class GAN(object):

    def __init__(self,
                 input_dimension: tuple,
                 discriminator_conv_filters: list,
                 discriminator_conv_kernel_size: list,
                 discriminator_conv_strides: list,
                 discriminator_batch_norm_momentum,
                 discriminator_activation,
                 discriminator_dropout_rate,
                 discriminator_learning_rate,
                 z_dim,
                 generator_batch_norm_momentum,
                 generator_initial_dense_layer_size: tuple,
                 generator_activation,
                 generator_dropout_rate,
                 generator_upsample: list,
                 generator_conv_filters: list,
                 generator_conv_kernel_size: list,
                 generator_conv_strides: list,
                 generator_learning_rate
                 ):

        self.model = None
        self.generator = None
        self.discriminator = None
        self.input_dim = input_dimension
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

        self.z_dim = z_dim
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.n_layers_generator = len(generator_conv_filters)
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_learning_rate = generator_learning_rate
        self.epoch = 0
        self.d_losses = []
        self.g_losses = []

    def discriminator_model(self):

        # Obraz na wejsciu dyskryminatora
        discriminator_input = tf.keras.Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input

        # Stos warstw konwolucyjnych
        for i in range(self.n_layers_discriminator):
            x = tf.keras.layers.Conv2D(filters=self.discriminator_conv_filters[i],
                                       kernel_size=self.discriminator_conv_kernel_size[i],
                                       strides=self.discriminator_conv_strides[i],
                                       padding='same',
                                       name=f'discriminator_conv_{i}')(x)
            # Ewentualna normalizacja batchowa
            if self.discriminator_batch_norm_momentum and i > 0:
                x = tf.keras.layers.BatchNormalization(momentum=self.discriminator_batch_norm_momentum)(x)

            # Funkcja aktywacji
            x = tf.keras.layers.Activation(self.discriminator_activation)(x)

            # Dropout (przekształca cześć wartosci z warstwy wejsciowej na 0, ogranicza nadmierne doposowanie)
            if self.discriminator_dropout_rate:
                x = tf.keras.layers.Dropout(rate=self.discriminator_dropout_rate)(x)

        # Spłaszczenie do postaci wektora
        x = tf.keras.layers.Flatten()(x)

        # Warstwa gesta z fukcja aktywacji sigmoid, ktora przekształca wyjscie z wartswy Dense do zakresu od 0 do 1
        # Im bliżej 1 tym obraz jest uznawany z bardziej realistyczny
        discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        # Model pobiera obraz wejsciowy i zwraca liczbe z zakresu 0 do 1 jako ocene realistycznosci obrazu
        self.discriminator = tf.keras.Model(discriminator_input, discriminator_output)

    def generator_model(self):

        # Na wejsciu wektor o rozmiarze z_dim
        generator_input = tf.keras.Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input

        # Warstwa gesta
        x = tf.keras.layers.Dense(np.prod(self.generator_initial_dense_layer_size))(x)

        # Ewentualna normalizacja batchowa(partii)- stabilizacja uczenia sieci
        if self.generator_batch_norm_momentum:
            x = tf.keras.layers.BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)

        # Funkcja aktywacji
        x = tf.keras.layers.Activation(self.generator_activation)(x)

        # Przekształcenie do postaci tensora o wymiariach generator_initial_dense_layer_size
        x = tf.keras.layers.Reshape(self.generator_initial_dense_layer_size)(x)

        # Dropout
        if self.generator_dropout_rate:
            x = tf.keras.layers.Dropout(rate=self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):

            if self.generator_upsample[i] == 2:
                x = tf.keras.layers.UpSampling2D()(x)
                x = tf.keras.layers.Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    name='generator_conv_' + str(i),
                    kernel_initializer=self.weight_init
                )(x)
            else:

                x = tf.keras.layers.Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name='generator_conv_' + str(i),
                    kernel_initializer=self.weight_init
                )(x)

            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = tf.keras.layers.BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
                x = get_activation(self.generator_activation)(x)
            else:
                x = tf.keras.layers.Activation('tanh')(x)

            generator_output = x

            # Model przekształca wektor o pewnych wymiarach i generuje tensor o wymiarach zgodnych z wymiarami obrazu
            self.generator = tf.keras.Model(generator_input, generator_output)

    def discriminator_compile(self):

        self.discriminator.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def build_adversarial(self):

        # COMPILE DISCRIMINATOR
        self.discriminator_compile()

        # COMPILE THE FULL GAN
        set_trainable(self.discriminator, False)

        # Model, który łączy generator i dyskriminator
        # Na wejsciu wektor o rozmiarze z_dim podobnie jak dla generatora
        model_input = tf.keras.Input(shape=(self.z_dim,), name='model_input')

        # Na wyjsciu dostajemy liczbe z przedziału od 0 do 1 zwróconą przez dyskryminator, któremu przekazujemu
        # na wejsciu obraz stworzony przez generator na podstawie wektora wejsciowego
        model_output = self.discriminator(self.generator(model_input))
        self.model = tf.keras.Model(model_input, model_output)

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0004), loss='binary_crossentropy',
                           metrics=['accuracy'])

        set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        # Szkolenie na autentycznych obrazach (odpowiedz 1 valid)
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # Szkolenie na wygenerowanych obrazach (odpowiedz 0 fake)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):

        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        # Szkolenie generatora, przekazujemy losowy szum a odpowiedziami sa 1
        # Funkcja straty wylicza jak bardzo wygenerowany obraz
        g_loss, g_acc = self.model.train_on_batch(noise, valid)
        return g_loss, g_acc

    def generator_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(os.getcwd(), f"{run_folder}/images/sample_%d.png" % epoch))
        plt.close()

    def save_model(self):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        pkl.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))

    # def load_model(self):
    #    self.model.load(os.path.join(run_folder, 'model.h5'))
    #    self.discriminator.load(os.path.join(run_folder, 'discriminator.h5'))
    #    self.generator.load(os.path.join(run_folder, 'generator.h5'))

    def train(self, x_train, batch_size, epochs):

        for epoch in range(self.epoch, self.epoch + epochs):
            print(f"Epoka: {epoch}")
            d = self.train_discriminator(x_train, batch_size)
            g = self.train_generator(batch_size=batch_size)

            print(f"[D loss: ({d[0]})(R {d[1]}, F {d[2]})]\n"
                  f"[D acc: ({d[3]})(R {d[4]}, F {d[5]})]\n"
                  f"[G loss: {g[0]}]\n"
                  f"[G acc: {g[1]}]")

            self.d_losses.append(d)
            self.g_losses.append(g)
            self.generator_images(self.epoch)
            self.set_epoch_number()
            self.epoch += 1
            print('\n')
        self.save_model()
        self.model.save_weights(os.path.join(run_folder, f'weights/weights-{self.epoch}.h5'))
        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(run_folder, 'viz/discriminator.png'),
                                  show_shapes=True,
                                  show_layer_names=True)
        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png'),
                                  show_shapes=True,
                                  show_layer_names=True)

    def load_weights(self):
        print(f'Load weights from folder {run_folder}')
        self.model.load_weights(os.path.join(run_folder, 'weights/weights.h5'))

    def set_epoch_number(self):
        file_name = f'{run_folder}/epoch.txt'
        with open(file_name, 'w') as file:
            file.write(f'{self.epoch}')

    def load_model(self):
        self.load_weights()
        actual_epoch = get_epoch_number()
        print(f"Load epoch number {actual_epoch} from file")
        self.epoch = actual_epoch

    def plot_loss(self):

        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plt.plot([x[0] for x in self.d_losses], color='black', linewidth=0.5)
        plt.plot([x[1] for x in self.d_losses], color='green', linewidth=0.5)
        plt.plot([x[2] for x in self.d_losses], color='red', linewidth=0.5)
        plt.plot([x[0] for x in self.g_losses], color='orange', linewidth=0.5)

        plt.legend(('Discriminator loss', 'Discriminator loss real', 'Discriminator loss fake',
                    'Generator loss'), fontsize=20)
        plt.title('Loss function', fontsize=20)

        plt.xlabel('batch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlim(0, len(self.d_losses))
        plt.ylim(0, 2)

        fig.savefig(os.path.join(os.getcwd(), f"{run_folder}/plots/loss_{self.epoch}.png"))

    def plot_accuracy(self):

        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plt.plot([x[3] for x in self.d_losses], color='black', linewidth=0.5)
        plt.plot([x[4] for x in self.d_losses], color='green', linewidth=0.5)
        plt.plot([x[5] for x in self.d_losses], color='red', linewidth=0.5)
        plt.plot([x[1] for x in self.g_losses], color='orange', linewidth=0.5)

        plt.legend(('Discriminator accuracy', 'Discriminator accuracy real', 'Discriminator accuracy fake',
                    'Generator accuracy'), fontsize=20)
        plt.title('Accuracy function', fontsize=20)

        plt.xlabel('batch', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlim(0, len(self.d_losses))
        plt.ylim(0, 2)

        fig.savefig(os.path.join(os.getcwd(), f"{run_folder}/plots/accuracy_{self.epoch}.png"))
