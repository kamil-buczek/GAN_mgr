# External
import matplotlib.pyplot as plt
import numpy as np


def show_sample_images(images, labels, labels_names: dict):
    rows = 5
    cols = 5

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    cnt = 0

    for i in range(rows):
        for j in range(cols):
            image = images[cnt, :, :, :]
            #  image = (image + 1) / 2.0
            image = np.clip(image, 0, 1)
            label = labels[cnt]
            label_str = labels_names[label]
            axs[i, j].set_title(f'({label}): {label_str}')
            axs[i, j].imshow(np.squeeze(image), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.set_facecolor('white')
    plt.show()