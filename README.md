## TransportNet

Code supplement for the neural network architecture using the relaxation approach of transport equations

### Usage

1. Create a python virtual environment (pyenv or conda). If you are using no virtual environment, please be aware of
   version incompatibilities of the installed packages and your own.
2. Install the project requirements (example for pip): ```pip install -r requirements.txt```
3. Choose an implementation (right now, only pytorch works, so please choose pytorch): ```cd pytorch```
4. Run the testcase by executing ```sh test_swissroll.sh``` or ```sh test_mnist.sh```

### Testcases

1. The swissroll testcase is a regression test, the resulting predictions after each epoch are saved
   in ```pytorch/results/model_x```.
2. MNIST is the classical handwritten digit classification datasets

### Models

1. Model 0: Naive implicit layer (does not work right now)
2. Model 1: Baseline explicit Resnet
3. Model 2: Fully nonlinear implicit layer using a Newton optimizer for rootfinding
4. Model 3: TransNet (Layerwise splitting, Splitting version1)
5. Model 4: TransNetSplit2 (Layerwise splitting, Splitting version2)
6. Model 5: TransNetSweeping (Networkwide sweeping, splitting version2) (TODO)

### Training parameters

We explain the flags for the python file or the batch-scripts

1. Layer size: -u
2. Batch size: -b
3. Epoch number: -e
4. Relaxation parameter (epsilon): -x
5. Time step dt: -d
6. Model type (see above): -m