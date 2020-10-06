from __future__ import print_function, division
from torch.utils.data import Dataset
import pickle
from predictors.utils.mlp_utils import get_arch_acc
class NasDataset(Dataset):
    """Nas bench 101 dataset."""

    def __init__(self, pickle_file = None, samplenum = 10000,sample = None,maxsize=8):
        """
        Args:=7
            pickle_file (string): Path to the pickle file containing graphs (adj, label, metrics).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if(pickle_file):
            with open(pickle_file, "rb") as f:
                self.graphs = pickle.load(f)
                # self.graphs = random.sample(self.graphs,samplenum)
        elif sample is not None:
            self.graphs = sample
        else:
            self.graphs = []
        self.maxsize = maxsize

    def __len__(self):
        return len(self.graphs)

    """process data """
    def __getitem__(self, graph_id):
        sample = get_arch_acc(self.graphs[graph_id],maxsize=self.maxsize)
        return sample


    def append_new_graph(self, new_graph):
        self.graphs += new_graph

    def get_dataset(self):
        return  self.graphs