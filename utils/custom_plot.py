import glob
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


save_root = os.path.join('.', 'debug')


class CustomPlot:
    def __init__(self):
        self.sequence_list_dict = {}

    def add_data(self, key, sequence):
        sequence = [float(value) for value in sequence]
        if key not in self.sequence_list_dict:
            self.sequence_list_dict[key] = []

        self.sequence_list_dict[key].append(sequence)

    def clear_plot(self):
        for f in glob.glob(os.path.join(save_root, '*.png')):
            os.remove(f)
        print('debug image cleared')

    def plot(self):
        for key in self.sequence_list_dict.keys():
            sequence_list = self.sequence_list_dict[key]
            sequence_list = np.array(sequence_list)
            sequence_mean = np.mean(sequence_list, axis=0)
            sequence_min = np.min(sequence_list, axis=0)
            sequence_max = np.max(sequence_list, axis=0)

            x = np.array([i for i in range(sequence_list.shape[1])])
            y = sequence_mean
            error_lower = sequence_mean - sequence_min
            error_upper = sequence_max - sequence_mean

            plt.errorbar(x, y, yerr=[error_lower, error_upper])
            plt.title(key)
            plt.savefig(os.path.join(save_root, '{}.png'.format(key)))
