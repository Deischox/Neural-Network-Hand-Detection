import numpy as np
import time
from layer import Layer
import matplotlib.pyplot as plt
import os

no_of_different_labels = 4
def forward(net, X):
    L = len(net)  # number of layers
    O = [None] * L  # list that collects the output tensors computed at each layer
    A = [None] * L  # list that collects the activation tensors computed at each layer

    for l in range(0, L):  # loop on layers
        if l == 0:
            A[l] = net[l].forward(X)
        else:
            A[l] = net[l].forward(O[l - 1])
        O[l] = net[l].activate(A[l])  # output of each layer
    return O, A


def loss(OL, y, compute_derivative=False):
    diff = OL - y
    if not compute_derivative:
        return np.mean(diff * diff)  # mean squared L2 norm (MSE)
    else:
        return (2.0 / y.shape[0]) * diff


def backward(net, A, d_loss):
    L = len(net)
    for l in range(L - 1, -1, -1):
        d_loss = net[l].backwards(A[l], d_loss)


def update(net, lr=0.001):
    for layer in net:
        layer.updateWeights(lr)

def onlineTraining():
    
    train_network = True
    save_as = "bp.npy"
    batch_size = 2

    # Define Data
    image_size = 28  # width and length
    image_pixels = image_size * image_size

    # Read Data from CSV File
    test_data = np.loadtxt("bp.csv",
                           delimiter=",")[-1:]
    fac = 0.99 / 255

    # include batch here
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    test_labels = np.asfarray(test_data[:, :1])
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    test_labels_one_hot = (lr == test_labels).astype(np.float64)

    # we don't want zeroes and ones in the labels neither:
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99


    # define the network
    my_net = np.load("bp.npy", allow_pickle=True)

    # training the network
    train(my_net, test_imgs, test_labels_one_hot,
            epochs=100, lr=0.1, batch_size=1)
    np.save(save_as, my_net)

def train(net, X, Y, epochs=2000, lr=0.001, batch_size=200):
    """Train a neural network for multiple epochs."""

    for e in range(0, epochs):  # loop on epochs

        # create mini-batch
        randomizer = np.arange(batch_size)
        np.random.shuffle(randomizer)
        X = X[randomizer]
        Y = Y[randomizer]
        outputs, activation_scores = forward(net, X)  # going forward
        loss_value = loss(outputs[-1], Y)

        print("epoch: {}, loss: {}".format(e + 1, loss_value))

        loss_derivative = loss(
            outputs[-1], Y, compute_derivative=True)  # going backward
        backward(
            net, activation_scores, loss_derivative)

        update(net, lr)  # updating model parameters


def predictDrawing(data):
    fac = 0.99 / 255
    if not os.path.isfile("bp.npy"):
        my_net = [Layer(784, 16), Layer(16, 16), Layer(16, no_of_different_labels)]
        np.save("bp.npy", my_net)
    test_imgs = np.asfarray(data) * fac + 0.01
    my_net = np.load('bp.npy', allow_pickle=True)
    net_outputs, _ = forward(my_net, test_imgs)
    print(net_outputs[-1])

# entry point
if __name__ == "__main__":

    train_network = False
    save_as = "drawing.npy"
    batch_size = 2

    # Define Data
    image_size = 28  # width and length
    image_pixels = image_size * image_size

    # Read Data from CSV File
    test_data = np.loadtxt("drawing.csv",
                           delimiter=",")
    fac = 0.99 / 255

    # include batch here
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    test_labels = np.asfarray(test_data[:, :1])
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    test_labels_one_hot = (lr == test_labels).astype(np.float64)

    # we don't want zeroes and ones in the labels neither:
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99

    duration = None
    if train_network:
        # define the network
        my_net = [Layer(image_pixels, 16), Layer(16, 16), Layer(16, no_of_different_labels)]
        net_outputs, _ = forward(my_net, test_imgs)
        # training the network
        start = time.time()
        train(my_net, test_imgs, test_labels_one_hot,
              epochs=10000, lr=0.1, batch_size=2)
        duration = time.time()-start
        np.save(save_as, my_net)
    my_net = np.load('drawing.npy', allow_pickle=True)

    """
    Results:

    learning rate = 0.1

    epoch: 10000 
    batch size: 10
    time: 5.4 sec
    Accuracy: 21%

    epoch: 10000
    batch size: 100
    time: 11 sec
    Accuracy: 47%

    epoch: 10000 
    batch size: 1000
    time: 72 sec
    Accuracy: 68%

    epoch: 10000 
    batch size: 10000
    time: 809 sec
    Accuracy: 77%

    
    epoch: 20000

    epoch: 20000 
    batch size: 10
    time: 10 sec
    Accuracy: 23%

    epoch: 20000
    batch size: 100
    time: 23 sec
    Accuracy: 49%

    epoch: 20000 
    batch size: 1000
    time: 147 sec
    Accuracy: 78%

    epoch: 20000 
    batch size: 10000
    time: X sec
    Accuracy: X%

    epoch: 50000 
    batch size: 10000
    time: 44 Minuten
    Accuracy: 93%
    -> net.npy

    lr = 0.5
    epoch = 20000
    batch_size = 500
    => time: 82 sec, Acc: 73%


    epoch=50000
    lr=0.1
    batch_size=1000
    => time: 328 sec Acc: 79%

    epoch=500000
    lr=0.1
    batch_size=1000
    => time: 3060 sec Acc: 79.48%



    """

    # making predictions with the trained network
    net_outputs, _ = forward(my_net, test_imgs)

    def helperPrediction(output_array, prediction_array):
        return output_array.argmax(), prediction_array.argmax()

    correct = 0
    # comparing predictions and expected targets
    for i in range(0, test_imgs.shape[0]):
        print(net_outputs[-1][i])
        e, p = helperPrediction(net_outputs[-1][i], test_labels_one_hot[i])
        if e == p:
            correct += 1
        print("input_id: {}, net_output: {}, "
              "expected_output: {}".format(i + 1, e, p))
    print(correct/test_imgs.shape[0], duration)
