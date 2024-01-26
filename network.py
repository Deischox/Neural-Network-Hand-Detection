from random import random

import numpy as np
import time
from layer import Layer
import matplotlib.pyplot as plt
import os

no_of_different_labels = 8
npy_file_name = "bp.npy"
LABELS = ["House", "Car", "Inear headphones", "Bottle",
          "On ear headphones", "Stick man", "TV-screen", "Sun"]


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


def softmax(OL):
    for i in range(len(OL)):
        OL[i] /= sum(OL[i])
    return OL


def loss(OL, y, compute_derivative=False):
    """
        Compute the cross-entropy loss.

        Parameters:
        OL (numpy.ndarray): Predicted probabilities for each class.
                                Shape: (number of samples, number of classes).
        y (numpy.ndarray): True labels in one-hot encoded form.
                                Shape: (number of samples, number of classes).

        Returns:
        float: Mean cross-entropy loss over the batch.
        """
    if not compute_derivative:
        # Small constant to prevent division by zero and log(0)
        epsilon = 1e-12
        y_pred = np.clip(OL, epsilon, 1. - epsilon)
        N = y_pred.shape[0]

        # Compute cross-entropy loss
        ce_loss = -np.sum(y * np.log(y_pred + epsilon)) / N
        return ce_loss
    else:
        return OL - y

    diff = OL - y
    if not compute_derivative:
        # mean squared L2 norm (MSE) TODO Chane loss function for classification
        return np.mean(diff * diff)
    else:
        return (2.0 / y.shape[0]) * diff


def backward(net, A, d_loss):
    L = len(net)
    for l in range(L - 1, -1, -1):
        d_loss = net[l].backwards(A[l], d_loss)


def update(net, lr=0.001):
    for layer in net:
        layer.updateWeights(lr)
# Returns current online training example and random element from each other class


def get_random_element_from_each_class():
    class_labels = np.arange(0, 8)
    csv_file = np.loadtxt("bp.csv", delimiter=",")
    new_example = csv_file[-1:]
    training_data = np.asarray(new_example)
    label_of_new_example = int(new_example[0, 0])
    class_labels_without_new_example = class_labels[class_labels !=
                                                    label_of_new_example]
    for clazz in class_labels_without_new_example:
        temp = [row for row in csv_file if int(row[0]) == clazz]
        index = np.random.randint(0, len(temp))
        random_element = temp[index]
        training_data = np.append(training_data, random_element)

    # return array
    return training_data.reshape(no_of_different_labels, 785)


def onlineTraining(last_index):

    save_as = npy_file_name
    batch_size = no_of_different_labels

    # Define Data
    image_size = 28  # width and length
    image_pixels = image_size * image_size

    # TODO remve
    print_after = False
    # Read Data from CSV File
    if last_index % 2 == 0:
        print_after = True
        training_data = get_random_element_from_each_class()
    else:
        print_after = False
        training_data = np.loadtxt("bp.csv",  # test_data is only last row
                                   delimiter=",")[-1:]
    fac = 0.99 / 255

    # include batch here
    training_imgs = np.asfarray(training_data[:, 1:]) * fac + 0.01
    training_labels = np.asfarray(training_data[:, :1])
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    training_labels_one_hot = (lr == training_labels).astype(np.float64)

    # we don't want zeroes and ones in the labels neither:
    training_labels_one_hot[training_labels_one_hot == 0] = 0.01
    training_labels_one_hot[training_labels_one_hot == 1] = 0.99

    # define the network
    my_net = np.load(npy_file_name, allow_pickle=True)

    # training the network
    train(my_net, training_imgs, training_labels_one_hot,
          epochs=100, lr=0.1, batch_size=len(training_imgs))  # TODO lr seems pretty large as well in general?
    np.save(npy_file_name, my_net)
    if print_after:
        print("next is after batch of other elements")


def train(net, X, Y, epochs=2000, lr=0.001, batch_size=200):
    """Train a neural network for multiple epochs."""

    for e in range(0, epochs):  # loop on epochs

        # create mini-batch
        randomizer = np.arange(batch_size)
        np.random.shuffle(randomizer)
        # X = X[randomizer]
        # Y = Y[randomizer]
        outputs, activation_scores = forward(net, X)  # going forward
        outputs = softmax(outputs[-1])
        loss_value = loss(outputs, Y)

        print("epoch: {}, loss: {}".format(e + 1, loss_value))

        loss_derivative = loss(
            outputs, Y, compute_derivative=True)  # going backward
        backward(
            net, activation_scores, loss_derivative)

        update(net, lr)  # updating model parameters

# correct solution:


def softmax_numpy(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def printPredictions(predictions):
    predictions = softmax_numpy(predictions)

    sorted_indices = np.argsort(predictions)[::-1]

    # Take the first three indices
    top_three = sorted_indices[:3]
    print(predictions)
    print(top_three)

    return f'{LABELS[top_three[0]]} {round(predictions[top_three[0]]*100,2)}% : {LABELS[top_three[1]]} {round(predictions[top_three[1]]*100,2)}% : {LABELS[top_three[2]]} {round(predictions[top_three[2]]*100,2)}%'


def predictDrawing(data):
    fac = 0.99 / 255
    if not os.path.isfile(npy_file_name):
        my_net = [Layer(784, 16), Layer(16, 16),
                  Layer(16, no_of_different_labels)]
        np.save(npy_file_name, my_net)
    test_imgs = np.asfarray(data) * fac + 0.01
    my_net = np.load(npy_file_name, allow_pickle=True)
    net_outputs, _ = forward(my_net, test_imgs)
    return printPredictions(net_outputs[-1])


# entry point
if __name__ == "__main__":

    train_network = True

    # Define Data
    image_size = 28  # width and length
    image_pixels = image_size * image_size

    # Read Data from CSV File
    test_data = np.loadtxt("bp.csv",
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
        my_net = [Layer(image_pixels, 16), Layer(16, 16),
                  Layer(16, no_of_different_labels)]
        # net_outputs, _ = forward(my_net, test_imgs)
        # training the network
        start = time.time()
        train(my_net, test_imgs, test_labels_one_hot,
              epochs=2000, lr=0.001, batch_size=100)
        duration = time.time()-start
        np.save(npy_file_name, my_net)
    # my_net = np.load(npy_file_name, allow_pickle=True)

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
        e, p = helperPrediction(net_outputs[-1][i], test_labels_one_hot[i])
        if e == p:
            correct += 1
    print(correct/test_imgs.shape[0], duration)
