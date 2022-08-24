# External
import matplotlib.pyplot as plt
import numpy as np
import os


def show_sample_images(images, labels, labels_names: dict, data_path, dataset_size: int):
    rows = 5
    cols = 5

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    cnt = 0

    random_samples = np.random.randint(0, dataset_size, 25)

    for i in range(rows):
        for j in range(cols):
            image = images[random_samples[cnt], :, :, :]
            #  image = (image + 1) / 2.0
            image = np.clip(image, 0, 1)
            label = labels[random_samples[cnt]]
            label_str = labels_names[label]
            axs[i, j].set_title(f'({label}): {label_str}')
            axs[i, j].imshow(np.squeeze(image), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.set_facecolor('white')
    fig.savefig(os.path.join(os.getcwd(), f"{data_path}_data/sample_images/sample_0_dataset.png"))
    plt.show()
