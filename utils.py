import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]