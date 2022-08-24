#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy as np

EPOCH = 300
BATCHES = EPOCH * 192

def load_loss_data_from_file(data_type: str, file_path: str) -> list:
    result_data: list = []

    file_path = f'{file_path}/.{data_type}'

    num = 1

    with open(file_path, 'r') as file:
        for line in file.readlines():
            if num > BATCHES:
                break
            result_data.append(float(line.strip()))
            num+=1

    return result_data


def plot_loss(network_names: list, data_type: str, plot_title: str, networks_path: str, width: float, height: float):

    plot_width = width
    plot_height = height

    fig = plt.figure(figsize=(15, 15), facecolor='white')

    legends = []

    for network in network_names:

        network_path = f'{networks_path}/{network}'
        network_loss_data: list = load_loss_data_from_file(data_type, network_path)

        plt.plot([_ for _ in network_loss_data], linewidth=0.2, )
        # plt.plot([_ for _ in disc_fake_data], color='green', linewidth=0.5)
        legends.append(network)

    leg = plt.legend(legends, fontsize=20, )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.title(plot_title, fontsize=15)

    plt.xlabel('batch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xticks(np.arange(0, width, 10000), fontsize=15)
    plt.yticks(np.arange(0, height, 0.5), fontsize=15)

    plt.xlim(0, plot_width)
    plt.ylim(0, plot_height)

    fig.savefig(os.path.join(os.getcwd(), f"plots/loss.{data_type}.png"))


if __name__ == "__main__":

    networks_path = '/home/kamil/Magisterka/time_plot'
    # networks_names = ['Basic_network', 'Network_No_Dropout', 'Network_Dropout_0.1', 'Network_Dropout_0.25', 'Network_Dropout_0.5']
    # networks_names = [f'{networks_path}/{_}' for _ in networks_names]
    networks_names = ['Network_LR_Both_0.0004', 'Basic_network', 'Network_LR_Both_0.00005',
                      'Network_LR_Discriminator_0.00005', 'Network_LR_Gan_0.00005']

    plot_loss(networks_names, 'loss_disc_fake', 'Wykresy funkcji straty dyskryminatora na fałszywych obrazach', networks_path, width=BATCHES, height=2.5)

    plot_loss(networks_names, 'loss_disc_real', 'Wykresy funkcji straty dyskryminatora na prawdziwych obrazach', networks_path, width=BATCHES, height=2.5)

    plot_loss(networks_names, 'loss_generator', 'Wykresy funkcji straty generatora', networks_path, width=BATCHES, height=7)