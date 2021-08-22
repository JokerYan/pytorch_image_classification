import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

class CustomPlot:
    def __init__(self):
        self.sequence_list_dict = {}

    def add_data(self, key, sequence):
        sequence = [float(value) for value in sequence]
        if key not in self.sequence_list_dict:
            self.sequence_list_dict[key] = []

        self.sequence_list_dict[key].append(sequence)

    def plot(self):
        for key in self.sequence_list_dict.keys():
            sequence_list = self.sequence_list_dict[key]
            sequence_list = np.array(sequence_list)
            sequence_mean = np.mean(sequence_list, axis=0)
            sequence_min = np.min(sequence_list, axis=0)
            sequence_max = np.max(sequence_list, axis=0)
            sequence_error_lower = sequence_mean - sequence_min
            sequence_error_upper = sequence_max - sequence_mean
            print(sequence_mean)
            print(sequence_error_upper)
            print(sequence_error_lower)
