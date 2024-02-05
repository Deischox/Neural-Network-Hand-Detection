
import numpy as np
import time
from activation import ReLU
from layer import Layer
import matplotlib.pyplot as plt
import os

no_of_different_labels = 8
npy_file_name = "models/model.npy"
LABELS = ["House", "Car", "Inear headphones", "Bottle",
          "On ear headphones", "Stick man", "TV-screen", "Sun"]


# Read Data from CSV File
eval_data = np.loadtxt("data/eval.csv",
                       delimiter=",")
fac = 0.99 / 255

# include batch here
eval_imgs = np.asfarray(eval_data[:, 1:]) * fac + 0.01
eval_labels = np.asfarray(eval_data[:, :1])


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
    csv_file = np.loadtxt("data/train.csv", delimiter=",")
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
    # Read Data from CSV File
    if last_index % 2 == 0:
        training_data = get_random_element_from_each_class()
    else:
        training_data = np.loadtxt("data/train.csv",  # test_data is only last row
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
          epochs=10, lr=0.01, batch_size=len(training_imgs), online=True)
    np.save(npy_file_name, my_net)


def train(net, X, Y, epochs=2000, lr=0.001, batch_size=200, online=False):
    """Train a neural network for multiple epochs."""

    loss_train = []
    loss_eval = []
    for e in range(0, epochs):  # loop on epochs

        # create mini-batch
        randomizer = np.arange(batch_size)
        np.random.shuffle(randomizer)
        X = X[randomizer]
        Y = Y[randomizer]
        # Eval
        outputs_eval, _ = forward(net, eval_imgs)  # going forward
        outputs_eval = softmax(outputs_eval[-1])
        loss_eval_value = loss(outputs_eval, eval_labels)
        loss_eval.append(loss_eval_value)

        outputs, activation_scores = forward(net, X)  # going forward
        outputs = softmax(outputs[-1])
        loss_value = loss(outputs, Y)
        loss_train.append(loss_value)

        print("epoch: {}, loss: {}".format(e + 1, loss_value))

        loss_derivative = loss(
            outputs, Y, compute_derivative=True)  # going backward
        backward(
            net, activation_scores, loss_derivative)

        update(net, lr)  # updating model parameters
    if not online:
        # Plot loss curves
        plt.plot(loss_train, label='Train Loss')
        plt.plot(loss_eval, label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def softmax_numpy(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def printPredictions(predictions):
    predictions = softmax_numpy(predictions)
    sorted_indices = np.argsort(predictions)[::-1]
    top_three = sorted_indices[:3]
    return f'{LABELS[top_three[0]]} {round(predictions[top_three[0]]*100,2)}% : {LABELS[top_three[1]]} {round(predictions[top_three[1]]*100,2)}% : {LABELS[top_three[2]]} {round(predictions[top_three[2]]*100,2)}%'


def predictDrawing(data):
    fac = 0.99 / 255
    if not os.path.isfile(npy_file_name):
        my_net = [Layer(784, 16, activation_function=ReLU()), Layer(16, 16, activation_function=ReLU()),
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
    test_data = np.loadtxt("data/train.csv",
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
              epochs=2000, lr=0.001, batch_size=30)
        duration = time.time()-start
        # np.save(npy_file_name, my_net)
    else:
        my_net = np.load(npy_file_name, allow_pickle=True)

    # making predictions with the trained network
    net_outputs, _ = forward(my_net, test_imgs)

    def helperPrediction(output_array, prediction_array):
        return output_array.argmax(), prediction_array.argmax()

    correct = 0
    wrong = 0
    # comparing predictions and expected targets
    for i in range(0, test_imgs.shape[0]):
        e, p = helperPrediction(net_outputs[-1][i], test_labels_one_hot[i])
        if e == p:
            correct += 1
        else:
            wrong += 1

    print(correct/test_imgs.shape[0], duration)
    print(correct, wrong)
