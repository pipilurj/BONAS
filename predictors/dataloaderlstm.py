from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
from predictors.utils.lstm_utils import get_arch_acc
random.seed(88)
class NasDataset(Dataset):
    """Nas bench 101 dataset."""

    def __init__(self, pickle_file = None, samplenum = 10000,sample = None,):
        """
        Args:
            pickle_file (string): Path to the pickle file containing graphs (adj, label, metrics).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if(pickle_file):
            with open(pickle_file, "rb") as f:
                self.graphs = pickle.load(f)
                self.graphs = random.sample(self.graphs,samplenum)
        elif sample is not None:
            self.graphs = sample
        else:
            self.graphs = []

    def __len__(self):
        return len(self.graphs)

    """process data """
    def __getitem__(self, graph_id):
        sample = get_arch_acc(self.graphs[graph_id])
        return sample

    def normalize_accuracy(self, accuracy):
        """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
        accuracy[accuracy < 0.8] = 0.8
        accuracy = (accuracy-0.8)/(0.94661456-0.8)
        accuracy = np.array(accuracy, dtype = np.float32)
        return accuracy

    def append_new_graph(self, new_graph):
        self.graphs += new_graph