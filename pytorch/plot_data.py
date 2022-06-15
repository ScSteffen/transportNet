import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from optparse import OptionParser

from src.bake_swiss_rolls import create_dataset
from src.utils import create_model, fit, test

if __name__ == '__main__':
    print("---------- Start Illustration Suite ------------")

    create_dataset(num_samples=200, plotting=True)
