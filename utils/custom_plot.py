import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

class CustomPlot:
    def __init__(self):
        self.sequence_list_dict = {}

    def add_data(self, key, sequence):
        sequence = list(sequence)
        if key not in self.sequence_list_dict:
            self.sequence_list_dict[key] = []

        self.sequence_list_dict[key].append(sequence)

    def plot(self):
        for key in self.sequence_list_dict.keys():
            sequence_list = self.sequence_list_dict[key]
            sequence_list = np.array(sequence_list)
            print(sequence_list.shape)
