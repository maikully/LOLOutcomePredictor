from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d
import csv

import os
import tensorflow as tf
import numpy as np
import random
import math


# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        def create_variable(dims):  # Easy initialization function for you :)
            return tf.Variable(tf.random.truncated_normal(dims, stddev=.1, dtype=tf.float32))

        self.b1 = create_variable([10])
        self.b2 = create_variable([5])
        self.b3 = create_variable([2])
        self.W1 = create_variable([10, 10])
        self.W2 = create_variable([10, 5])
        self.W3 = create_variable([5, 2])

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: games, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        dense1_output = tf.linalg.matmul(inputs, self.W1)
        bias1_output = tf.nn.bias_add(dense1_output, self.b1)
        dense2_output = tf.linalg.matmul(bias1_output, self.W2)
        bias2_output = tf.nn.bias_add(dense2_output, self.b2)
        dense3_output = tf.linalg.matmul(bias2_output, self.W3)
        bias3_output = tf.nn.bias_add(dense3_output, self.b3)
        return tf.nn.leaky_relu(bias3_output)

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels - no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    indices = tf.random.shuffle(
        tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32))
    train_inputs = tf.gather(train_inputs, indices)
    train_inputs = tf.image.random_flip_left_right(train_inputs)
    train_labels = tf.gather(train_labels, indices)
    losses = list()
    for i in range(0, math.ceil(len(train_inputs)/model.batch_size)):
        inputs_batch = train_inputs[i *
            model.batch_size: (i + 1) * model.batch_size]
        labels_batch = train_labels[i *
            model.batch_size: (i + 1) * model.batch_size]
        with tf.GradientTape() as tape:
            logits = model.call(inputs_batch)
            loss = model.loss(logits, labels_batch)
            losses.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    return losses


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    logits = model.call(test_inputs)
    # visualize_results(test_inputs[0:50], logits[0:50], test_labels[0:50], "cat", "dog")
    return model.accuracy(logits, test_labels)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            "{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def get_data(inputs_path, labels_path):
    inputs = []
    labels = []
    with open(inputs_path, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        # CHANGE DATA INTO ONE-HOT CATEGORICAL
        inputs.append(row)
        print(row)
    inputs = np.array(inputs).astype(np.float32)
    #inputs = tf.reshape(inputs, (-1, 3, 32 , 32))
    with open(labels_path, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        labels.append(row)
        print(row)
    labels = np.array(labels).astype(np.float32)
    return (inputs, labels)


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 

    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''
    (test_inputs, test_labels) = get_data("Champs_test.csv","Outcomes_test.csv")
    (train_inputs, train_labels) = get_data("Champs_train.csv","Outcomes_train.csv")
    model = Model()
    train_epochs = 25
    for i in range(train_epochs):
        loss_list = train(model, train_inputs, train_labels)
    print("testing accuracy: ")
    print(np.array(test(model, test_inputs, test_labels)))
    visualize_loss(loss_list)


if __name__ == '__main__':
    main()
