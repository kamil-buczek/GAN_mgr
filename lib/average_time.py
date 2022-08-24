#!/usr/bin/env python3

def get_training_time(network_path):
    time_file_path = f'{network_path}/.times'
    sum_in_sec = 0

    num_of_lines = 0

    with open(time_file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip().split('=')
            seconds = float(line[1])
            sum_in_sec += seconds
            num_of_lines += 1

    hours, rest = divmod(sum_in_sec, 3600)
    minutes, seconds = divmod(rest, 60)

    avg_time_sec = float(sum_in_sec / num_of_lines)

    print(f'-------------------- {network_path}----------------------------')
    print(f'----> Total training time: {hours}h {minutes}m, {int(seconds)}s')
    print(f'----> Number of loaded epochs: {num_of_lines}')
    print(f'----> Average time of training one epoch is: {avg_time_sec} second')


if __name__ == "__main__":

    networks_path = '/home/kamil/Magisterka/time_plot'
    # network_name2 = 'Basic_network'
    # network_name = 'Network_128x128'

    # networks_names = ['Basic_network', 'Network_No_Dropout', 'Network_Dropout_0.1', 'Network_Dropout_0.25', 'Network_Dropout_0.5']
    networks_names = ['Basic_network', 'Network_LR_Both_0.001', 'Network_LR_Both_0.0004', 'Network_LR_Both_0.00005',
                      'Network_LR_Discriminator_0.00005', 'Network_LR_Gan_0.00005']

    for _ in networks_names:
        get_training_time(f'{networks_path}/{_}')
    #get_training_time(f'{networks_path}/{network_name2}')