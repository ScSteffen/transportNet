from src.networks.transportNetImplicit import TransNetImplicit, create_csv_logger_cb

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

import numpy as np
from optparse import OptionParser
from os import path, makedirs

import swiss_roll_testcase.bake_swiss_rolls as bake_swiss_rolls

from src_torch.networks.simple_implicit import ImplicitNet, ImplicitLayer


def train(num_layers, units, epsilon, batch_size, load_model, epochs):
    """
    :param num_layers:
    :param units:
    :param epsilon:
    :param batch_size:
    :param load_model:
    :param epochs:
    :return: Neural network trianing suite
    """

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 0) Sanitychecks (gradient works!)
    # from torch.autograd import gradcheck
    # layer = ImplicitLayer(size_in=2, size_out=2)
    # gradcheck(layer, torch.randn(3, 2, requires_grad=True, dtype=torch.double), check_undefined_grad=False)

    # 1) Create network
    model = ImplicitNet(units=units, input_dim=2, output_dim=2).to(device)
    print(model)

    # 2)  Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 3) Create Dataset

    (x_train, y_train), (x_test, y_test) = bake_swiss_rolls.create_dataset()

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train.reshape(y_train.shape[0], )).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    tensor_x = torch.Tensor(x_test)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test.reshape(y_test.shape[0], )).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    test_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [batch, dim]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # train the network
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        fit(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # 3) Call network
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    return 0


def fit(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Sanity checks
        # print(model.linear_relu_stack._modules['0'].A)

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-u", "--units", dest="units", default=200)
    parser.add_option("-x", "--epsilon", dest="epsilon", default=0.1)
    parser.add_option("-l", "--load_model", dest="load_model", default=1)
    parser.add_option("-t", "--train", dest="train", default=1)
    parser.add_option("-b", "--batch_size", dest="batch_size", default=32)
    parser.add_option("-e", "--epochs", dest="epochs", default=100)
    parser.add_option("-n", "--num_layers", dest="num_layers", default=100)

    (options, args) = parser.parse_args()
    options.units = int(options.units)
    options.epsilon = float(options.epsilon)
    options.load_model = int(options.load_model)
    options.train = int(options.train)
    options.batch_size = int(options.batch_size)
    options.epochs = int(options.epochs)
    options.num_layers = int(options.num_layers)

    if options.train == 1:
        train(num_layers=options.num_layers, units=options.units, epsilon=options.epsilon,
              batch_size=options.batch_size, load_model=options.load_model, epochs=options.epochs)
