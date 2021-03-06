from src.simple_implicit import ImplicitNet, ImplicitLayer
from src.resnet import ResNet
from src.newton_implicit import NewtonImplicitNet
from src.transNetImplicit import TransNetImplicit, TransNetLayer
from src.transNetExplicit import TransNetExplicit
from src.transNetExplicit2 import TransNetExplicit2
from src.transNetImplicit_split2 import TransNetSplit2, TransNetLayerSplit2
from src.transNetImplicitSweeping import TransNetSweeping
from src.transNetImplicitSweeping_explicitrhs import TransNetSweepingExplRhs
from torch.autograd import gradcheck
import torch

from os import path, makedirs
from timeit import default_timer as timer


def create_model(model_type: int = 0, units: int = 10, num_layers: int = 4, device="cpu", input_dim: int = 784,
                 output_dim: int = 10, dt: float = 0.1, epsilon: float = 0.01, grad_check: bool = False,
                 batch_size: int = 32, steps=40):
    """
    :param model_type: Type of the created model
    :param units: width of each (implicit) resnet layer
    :param num_layers: number of layers ( currently hardcoded)
    :param device: computational device
    :param input_dim: dimension of linear input layer
    :param output_dim: dimension of linear output layer
    :param dt: timestep of relaxation model
    :param epsilon: relaxation tolerance
    :param grad_check: if gradient should be checked in model
    :param batch_size: batch size for training
    :return: created model
    """

    if model_type == 0:
        model = ImplicitNet(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers).to(device)
        print("implicitNet chosen")
        layer = ImplicitLayer(in_features=2, out_features=2)
        # gcheck = gradcheck(layer, torch.randn(batch_size, units, requires_grad=True, dtype=torch.float),
        #                   check_undefined_grad=False,
        #                   atol=1e-7)
        # if gcheck:
        #    print("Gradient of implicit layer corresponds to gradient of finite difference approximation")

    if model_type == 1:
        model = ResNet(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                       device=device).to(device)
        print("explicit ResNet chosen")

    if model_type == 2:
        model = NewtonImplicitNet(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                  device=device).to(
            device)
        print("Nonlinear implicit ResNet chosen")

    if model_type == 3:
        model = TransNetImplicit(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                 epsilon=epsilon, dt=dt,
                                 device=device).to(device)  # .double()
        print("TransNet chosen")
        # layer = TransNetLayer(in_features=units, out_features=units).double()
        # gcheck = gradcheck(layer, torch.randn(batch_size, 2 * units, requires_grad=True, dtype=torch.double),
        #                   check_undefined_grad=False, atol=1e-7)
        # if gcheck:
        #    print("Gradient of implicit layer corresponds to gradient of finite difference approximation")

    if model_type == 4:
        model = TransNetSplit2(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                               epsilon=epsilon, dt=dt,
                               device=device).to(device)  # .double()
        print("TransNet chosen wih v2 splitting")
        # layer = TransNetLayerSplit2(in_features=units, out_features=units).double()
        # gcheck = gradcheck(layer, torch.randn(batch_size, 2 * units, requires_grad=True, dtype=torch.double),
        #                   check_undefined_grad=False, atol=1e-7)
        # if gcheck:
        #    print("Gradient of implicit layer corresponds to gradient of finite difference approximation")

    if model_type == 5:
        model = TransNetSweeping(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                 epsilon=epsilon, dt=dt, device=device, steps=steps).to(device)  # .double()
        print("TransNet with sweeping and implicit right hand side chosen")

    if model_type == 6:
        model = TransNetSweepingExplRhs(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                        epsilon=epsilon, dt=dt, device=device, steps=steps).to(device)  # .double()
        print("TransNet with sweeping and explicit right hand side chosen")

    if model_type == 7:
        model = TransNetExplicit(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                 epsilon=epsilon,
                                 dt=dt, device=device).to(device)
        print("Explicit TransNet chosen")

    if model_type == 8:
        model = TransNetExplicit2(units=units, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                                  epsilon=epsilon,
                                  dt=dt, device=device).to(device)
        print("Explicit TransNet chosen with v2 splitting")
    # print(model)
    # 0) Sanitycheck
    if grad_check:
        model.double()
        gcheck = gradcheck(model, torch.randn(batch_size, input_dim, requires_grad=True, dtype=torch.float),
                           check_undefined_grad=False, atol=1e-7)
        if gcheck:
            print("Gradient of model corresponds to gradient of finite difference approximation")
        model.float()
    return model


def fit(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    # if sweeping_model:
    #    for batch, (X, y) in enumerate(dataloader):
    #        X, y = X.to(device), y.to(device)
    #        z = torch.flatten(X, start_dim=1)
    #        model.initialize_model(z)
    #        break

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Time measure start
        start = timer()
        # Compute prediction error
        z = torch.flatten(X, start_dim=1)
        pred = model(z)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        end = timer()
        timing = end - start
        # gradient update
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return timing


def test(dataloader, model, loss_fn, device, iter, log_file, timing):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # if sweeping_model:
    #    for batch, (X, y) in enumerate(dataloader):
    #        X, y = X.to(device), y.to(device)
    #        z = torch.flatten(X, start_dim=1)
    #
    #        model.initialize_model(z)
    #        break

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

    if iter == 1:
        data = ["iter", "Accuracy", "Loss", "timing"]
        write_history_log(log_file, data)

    data = [iter, correct, test_loss, timing]
    write_history_log(log_file, data)


def create_csv_logger_cb(folder_name: str):
    '''
    dynamically creates a csvlogger and tensorboard logger
    '''
    # check if dir exists
    if not path.exists(folder_name + '/historyLogs/'):
        makedirs(folder_name + '/historyLogs/')

    # checkfirst, if history file exists.
    logName = folder_name + '/historyLogs/history_001_'
    count = 1
    while path.isfile(logName + '.csv'):
        count += 1
        logName = folder_name + \
                  '/historyLogs/history_' + str(count).zfill(3) + '_'

    logFileName = logName + '.csv'
    # create logger callback
    f = open(logFileName, "a")

    return f, logFileName


def write_history_log(file: str, data: list):
    log_string = ""
    for pt in data:
        log_string += str(pt) + ";"
    log_string += "\n"

    with open(file, "a") as log:
        log.write(log_string)

    return 0
