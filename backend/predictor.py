from __future__ import absolute_import
from matplotlib import pyplot as plt
import csv

import os
import tensorflow as tf
import numpy as np
import random
import math
import json
from tqdm import trange, tqdm 

from sklearn.model_selection import KFold

# ensures that we run only on cpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Model(tf.keras.Model):
    def __init__(self):
        """
        Model containing architecture for the neural network
        """
        super(Model, self).__init__()

        self.batch_size = 200
        self.num_classes = 2
        self.total_champs = 162
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        def create_variable(dims):  # Easy initialization function for you :)
            return tf.Variable(tf.random.truncated_normal(dims, stddev=.1, dtype=tf.float32))

        self.Dense1 = tf.keras.layers.Dense(
            300, activation=tf.keras.layers.LeakyReLU())
        self.Dense2 = tf.keras.layers.Dense(
            1000, activation=tf.keras.layers.LeakyReLU())
        self.Dense3 = tf.keras.layers.Dense(
            200, activation=tf.keras.layers.LeakyReLU())
        self.Dense4 = tf.keras.layers.Dense(
            300, activation=tf.keras.layers.LeakyReLU())
        self.Dense5 = tf.keras.layers.Dense(
            200, activation=tf.keras.layers.LeakyReLU())
        self.Dense6 = tf.keras.layers.Dense(
            100, activation=tf.keras.layers.LeakyReLU())
        self.Dense7 = tf.keras.layers.Dense(
            2, activation=tf.keras.layers.LeakyReLU())
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.dropout = tf.keras.layers.Dropout(rate=.3)
        self.dropout1 = tf.keras.layers.Dropout(rate=.2)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of matches

        :param inputs: games, shape of (num_inputs, total_champs * 2, 1); during training, the shape is (batch_size, total_champs * 2, 1)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        #dropout layer?
        out = self.dropout(self.Dense1(inputs))
        out = self.dropout(self.Dense2(out))
        #out = self.dropout(self.Dense5(out))
        #out = self.dropout(self.Dense6(out))
        out = self.Dense7(out)
        #print(self.softmax(out))
        return out

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes) containing the resulting 
        predictions
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        #print(logits)
        #print(labels)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        #print(loss)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing logits to correct labels

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        #print(tf.argmax(logits, 1))
        #print(tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch, where inputs and labels are shuffled.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, total_champs, 2)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    indices = tf.random.shuffle(
        tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32))
    train_inputs = tf.gather(train_inputs, indices)
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
            print(loss)
            losses.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    return losses


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.

    :param test_inputs: test data (matches, shape (num_inputs, total_champs, 2))
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy
    """
    logits = model.call(test_inputs)
    loss = model.loss(logits, test_labels)
    print("testing loss: ")
    print(loss)
    #visualize_results(test_inputs[0:50], logits[0:50], test_labels[0:50], "1", "2")
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


def get_data(inputs_path, labels_path, key_to_encoding, total_champs):
    inputs = []
    labels = []
    counter = {}
    max = 0
    most_champ = 0
    win_counts = [0,0]
    with open(inputs_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(reader):
            # skip first line
            if i != 0:
                # CHANGE DATA INTO ONE-HOT CATEGORICAL
                champs = [int(x) for x in row[0].split(",")]
                arr = [0] * (total_champs * 2)
                for champ in champs[0:5]:
                    if champ in counter.keys():
                        counter[champ] += 1
                        if counter[champ] > max:
                            max = counter[champ]
                            most_champ = champ
                    else:
                        counter[champ] = 1
                    arr[key_to_encoding[champ]] = 1
                for champ in champs[5:10]:
                    if champ in counter.keys():
                        counter[champ] += 1
                        if counter[champ] > max:
                            max = counter[champ]
                            most_champ = champ
                    else:
                        counter[champ] = 1
                    arr[key_to_encoding[champ] + total_champs] = 1
                inputs.append(arr)
    # make inputs (num_inputs x total_champs x 2)
    inputs = np.array(inputs).astype(np.float32)
    #inputs = tf.reshape(inputs, (-1, 3, 32 , 32))
    with open(labels_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(reader):
            if i != 0:
                #print([int(x) for x in row[0].split(",")])
                win_counts[np.argmax([int(x) for x in row[0].split(",")])] += 1
                labels.append([int(x) for x in row[0].split(",")])
    labels = np.array(labels).astype(np.float32)
    print("most common champion (riot key): " + str(most_champ))
    print("win counts: " + str(win_counts))
    return (inputs, labels)


def make_one_hot(total_champs, encodings):
    """
    Turns an encodings matrix into data readable by the model.
    :param encodings: encodings matrix, shape (10)
    :return: matrix representing the match with one-hot encoding, shape (total_champs, 2)
    """
    arr = np.zeros((total_champs, 2))
    for encoding in encodings[0:5]:
        arr[encoding][0] = 1
    for encoding in encodings[5:10]:
        arr[encoding][1] = 1
    return arr


def create_champ_map():
    f = open('../data/champion.json')

    key_to_encoding = {}
    encoding_to_key = {}
    name_to_encoding = {}
    data = json.load(f)
    for i, key in enumerate(data["data"].keys()):
        encoding_to_key[int(i)] = int(data["data"][key]["key"])
        key_to_encoding[int(data["data"][key]["key"])] = int(i)
        name_to_encoding[data["data"][key]["name"].lower()] = int(i)
    # Closing file
    f.close()
    return (key_to_encoding, encoding_to_key, name_to_encoding)


def main():
    '''
    Reads in matches data, initializes model, trains for n epochs, and 
    tests.

    :return: None
    '''
    model = Model()
    (key_to_encoding, encoding_to_key, name_to_encoding) = create_champ_map()
    (test_inputs, test_labels) = get_data("../data/Champs_test.csv",
                                          "../data/Outcomes_test.csv", key_to_encoding, model.total_champs)
    (train_inputs, train_labels) = get_data("../data/Champs_train.csv",
                                            "../data/Outcomes_train.csv", key_to_encoding, model.total_champs)
    
    train_epochs = 10
    print("training model...")
    loss_list = []
    #(X, Y) = get_data("../data/Champs.csv","../data/Outcomes.csv", key_to_encoding, model.total_champs)
    kf = KFold(n_splits=5)
    evaluations = []
    """
    for train_index, test_index in kf.split(X):
        model = Model()
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        loss_list.append(train(model, X_train, Y_train))
        #print("testing accuracy: ")
        #print(np.array(test(model, X_test, Y_test)))
        evaluations.append(np.array(test(model, X_test, Y_test)))
    """
    
    for i in trange(train_epochs):
        loss_list.append(train(model, train_inputs, train_labels))
    print("training finished!")
    print("testing accuracy: ")
    print(np.array(test(model, test_inputs, test_labels)))
    #visualize_loss(loss_list)
    #print(tf.reduce_mean(evaluations))
    """
    while (True):
        champs = input("enter the names of 10 champions, separated by commas.\nex: Amumu,xin zhao,akali, Renekton, Kha'Zix,rek'sai, etc.")
        encodings = [name_to_encoding(x.lower().strip()) for x in champs.split(",")]
        input = make_one_hot(model.total_champs, encodings)
        prediction = model.call(input)
        winner_prob = max(prediction)
        winner = tf.argmax(prediction) + 1
        print("The model predicts that team " + str(winner) +
              " will win with probability " + str(winner_prob) + ".")
              """


if __name__ == '__main__':
    main()
