import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

from optparse import OptionParser

from src.simple_implicit import ImplicitNet, ImplicitLayer
from src.resnet import ResNet
from src.newton_implicit import NewtinImplictNet
from src.transNetImplicit import TransNet, TransNetLayer

from torch.autograd import gradcheck


def train(num_layers, units, epsilon, batch_size, load_model, epochs, model_type):
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
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 1) Create network
    if model_type == 0:
        model = ImplicitNet(units=units, input_dim=784, output_dim=10, num_layers=num_layers).to(device).double()
        print("implicitNet chosen")
        layer = ImplicitLayer(in_features=2, out_features=2)
        # gcheck = gradcheck(layer, torch.randn(batch_size, units, requires_grad=True, dtype=torch.float),
        #                   check_undefined_grad=False,
        #                   atol=1e-7)
        # if gcheck:
        #    print("Gradient of implicit layer corresponds to gradient of finite difference approximation")

    if model_type == 1:
        model = ResNet(units=units, input_dim=784, output_dim=10, num_layers=num_layers).to(device).double()
        print("explicit ResNet chosen")

    if model_type == 2:
        model = NewtinImplictNet(units=units, input_dim=784, output_dim=10, num_layers=num_layers).to(device).double()

    if model_type == 3:
        model = TransNet(units=units, input_dim=784, output_dim=10, num_layers=num_layers).to(device).double()
        print("TransNet chosen")
        layer = TransNetLayer(in_features=units, out_features=units).double()
        # gcheck = gradcheck(layer, torch.randn(batch_size, 2 * units, requires_grad=True, dtype=torch.double),
        #                   check_undefined_grad=False, atol=1e-7)
        # if gcheck:
        #    print("Gradient of implicit layer corresponds to gradient of finite difference approximation")

    # print(model)
    # 0) Sanitycheck
    #gcheck = gradcheck(model, torch.randn(batch_size, 784, requires_grad=True, dtype=torch.double),
    #                   check_undefined_grad=False, atol=1e-7)
    #if gcheck:
    #    print("Gradient of model corresponds to gradient of finite difference approximation")

    # 2)  Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters())
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

        X = X.double()
        # y = y.double()
        # Compute prediction error
        z = torch.flatten(X, start_dim=1)
        pred = model(z)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Sanity checks
        # print(model.linear_relu_stack._modules['0'].A)

        # model.print_grads()
        # optimization
        # model.print_weights()
        optimizer.step()
        # model.print_weights()

        optimizer.zero_grad()
        # print("------")

        # for name, param in model.named_parameters():
        #    print(name, param)
        #    break

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
            X, y = X.to(device).double(), y.to(device)
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
    parser.add_option("-m", "--model_type", dest="model_type", default=0)

    (options, args) = parser.parse_args()
    options.units = int(options.units)
    options.epsilon = float(options.epsilon)
    options.load_model = int(options.load_model)
    options.train = int(options.train)
    options.batch_size = int(options.batch_size)
    options.epochs = int(options.epochs)
    options.num_layers = int(options.num_layers)
    options.model_type = int(options.model_type)

    if options.train == 1:
        train(num_layers=options.num_layers, units=options.units, epsilon=options.epsilon,
              batch_size=options.batch_size, load_model=options.load_model, epochs=options.epochs,
              model_type=options.model_type)
