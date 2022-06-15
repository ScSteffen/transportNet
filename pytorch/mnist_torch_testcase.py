import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

from optparse import OptionParser

from src.utils import create_model, fit, test


def train(num_layers, units, epsilon, dt, batch_size, load_model, epochs, model_type):
    """
    :param num_layers: numbre of layers
    :param units: neurons per layer
    :param epsilon: relaxation tolerance
    :param batch_size: batch size for training
    :param load_model: if model should be loaded from file (not yet implemented)
    :param epochs: number of training epochs
    :return: Neural network trianing suite
    """

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print("model type")
    print(model_type)

    # 1) Create network
    model = create_model(model_type=model_type, units=units, num_layers=num_layers, device=device, input_dim=784,
                         output_dim=10, dt=dt, epsilon=epsilon, grad_check=False, batch_size=batch_size)
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
    parser.add_option("-d", "--time_step", dest="dt", default=1)

    (options, args) = parser.parse_args()
    options.units = int(options.units)
    options.epsilon = float(options.epsilon)
    options.load_model = int(options.load_model)
    options.train = int(options.train)
    options.batch_size = int(options.batch_size)
    options.epochs = int(options.epochs)
    options.num_layers = int(options.num_layers)
    options.model_type = int(options.model_type)
    options.dt = float(options.dt)

    if options.train == 1:
        train(num_layers=options.num_layers, units=options.units, epsilon=options.epsilon,
              batch_size=options.batch_size, load_model=options.load_model, epochs=options.epochs,
              model_type=options.model_type, dt=options.dt)
