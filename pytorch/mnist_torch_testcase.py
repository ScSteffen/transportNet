import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

from optparse import OptionParser

from src.networks.simple_implicit import ImplicitNet, ImplicitLayer


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
    model = ImplicitNet(units=units, input_dim=784, output_dim=10).to(device)
    print(model)

    # 2)  Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 3) Create Dataset

    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # train the network
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        fit(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

        for name, param in model.named_parameters():
            print(name, param)
    print("Done!")

    # 3) Call network
    logits = model(torch.flatten(X, start_dim=1))
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
        z = torch.flatten(X, start_dim=1)
        pred = model(z)
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
            pred = model(torch.flatten(X, start_dim=1))
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
