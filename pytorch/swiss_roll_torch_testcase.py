import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from optparse import OptionParser

from src.bake_swiss_rolls import create_dataset
from src.utils import create_model, fit, test,create_csv_logger_cb

import numpy as np
import matplotlib.pyplot as plt


def train(num_layers, units, epsilon, dt, batch_size, load_model, epochs, model_type, plotting=False, gpu=True):
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
    if gpu==1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    print(f"Using {device} device")
    print("model type")
    print(model_type)

    # 1) Create network
    model = create_model(model_type=model_type, units=units, num_layers=num_layers, device=device, input_dim=2,
                         output_dim=2, dt=dt, epsilon=epsilon, grad_check=False, batch_size=batch_size)
    # 2)  Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # 3) Create Dataset

    (x_train, y_train), (x_test, y_test) = create_dataset(num_samples=10000)

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train.reshape(y_train.shape[0], )).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    tensor_x = torch.Tensor(x_test)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test.reshape(y_test.shape[0], )).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    test_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    n_grid = 100
    whole_space = np.zeros(shape=(n_grid, n_grid, 2))
    dx = 0.05
    c_i = 0
    c_j = 0
    for i in np.linspace(-1, 1, n_grid):
        for j in np.linspace(-1, 1, n_grid):
            whole_space[c_i, c_j, :] = np.asarray([i, j])
            c_j += 1
        c_i += 1
        c_j = 0
    whole_space_torch = torch.Tensor(np.reshape(whole_space, newshape=(n_grid ** 2, 2))).to(device)

    (x_train_plot, y_train_plot), _ = create_dataset(num_samples=200, test_rate=0.0, plotting=False,
                                                     shuffle=False)

    for X, y in test_dataloader:
        print(f"Shape of X [batch, dim]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    log_file, file_name = create_csv_logger_cb(folder_name="results/swiss_roll_model_" + str(model_type))


    # train the network
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        fit(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device,t+1,file_name)
        if plotting:
            print_current_evaluation(x_train_plot, whole_space_torch, n_grid, model, t + 1, model_nr=model_type)
    print("Done!")

    # 3) Call network
    logits = model(X.to(device))
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    return 0


def print_current_evaluation(train_x, whole_space_torch, n_grid, model, iter, model_nr):
    pred = model(torch.flatten(whole_space_torch, start_dim=1))
    pred_class = pred.argmax(1).cpu().numpy().reshape(n_grid, n_grid)

    z = pred_class
    c = plt.imshow(np.transpose(z), cmap='RdGy', vmin=0, vmax=1, extent=[-1, 1, -1, 1],
                   interpolation='none',
                   origin='lower')
    plt.colorbar(c)

    plt.title('prediction and training data', fontweight="bold")

    n_data = int(len(train_x) / 2)
    plt.plot(train_x[:n_data, 0], train_x[:n_data, 1], "r")
    plt.plot(train_x[n_data:, 0], train_x[n_data:, 1], "k")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))

    # plt.show()
    plt.savefig("results/swiss_roll_model_" + str(model_nr) + "/img_" + str(iter) + ".png", dpi=500)
    plt.clf()
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
    parser.add_option("-p", "--plotting", dest="plotting", default=1)
    parser.add_option("-g", "--gpu", dest="gpu", default=1)


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
    options.plotting = bool(options.plotting)
    options.gpu = int(options.gpu)


    if options.train == 1:
        train(num_layers=options.num_layers, units=options.units, epsilon=options.epsilon,
              batch_size=options.batch_size, load_model=options.load_model, epochs=options.epochs,
              model_type=options.model_type, dt=options.dt, plotting=options.plotting, gpu =  options.gpu)
