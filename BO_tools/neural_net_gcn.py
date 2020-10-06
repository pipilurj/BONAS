from predictors.models import GCN
from predictors.utils.gcn_train_val import train, validate
import torch
import torch.optim as optim
from predictors.dataloader import NasDataset
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import random


class NeuralNet(object):

    def __init__(self, dataset, val_dataset=None, ifPretrained=False, maxsize=7):
        """Initialization of NeuralNet object

        Keyword arguments:
        architecture -- a tuple containing the number of nodes in each layer
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        self.__val_dataset = val_dataset
        self.gcn = None
        self.ifPretrained = ifPretrained
        self.maxsize = maxsize

    def train(self, lr, num_epoch, selected_loss, ifsigmoid):
        dataset = NasDataset(sample=self.__dataset, maxsize=self.maxsize)
        val_dataset = NasDataset(sample=self.__val_dataset, maxsize=self.maxsize)
        batch_size = 128
        # Creating PT data samplers and loaders:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, )
        gcn = GCN(
            nfeat=len(self.__dataset[0]['operations'][0]) + 1,
            ifsigmoid=ifsigmoid
        )
        gcn = gcn.cuda()
        gcn = torch.nn.DataParallel(gcn)
        optimizer = optim.Adam(gcn.parameters(),
                               lr=lr,
                               )
        loss = selected_loss
        if self.__val_dataset:
            validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                            shuffle=True, num_workers=4, )
            for i in range(num_epoch):
                train(model=gcn, optimizer=optimizer, loss=loss, train_loader=train_loader, epoch=i)
                validate(model=gcn, loss=loss, validation_loader=validation_loader)
            self.gcn = gcn
        else:
            for i in range(num_epoch):
                train(model=gcn, optimizer=optimizer, loss=loss, train_loader=train_loader, epoch=i)
            self.gcn = gcn

    @property
    def network(self):
        return self.gcn