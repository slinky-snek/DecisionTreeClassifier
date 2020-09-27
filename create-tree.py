# Aaron Barnett
# acba242@uky.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

global original_data


class Node:
    def __init__(self):
        self.children = []  # index of children in node list
        self.prediction = -1  # is leaf, prediction if true
        self.feature = ''  # feature this node splits on


def entropy(data):
    entropy = 0
    num_samples = len(data)
    pos_samples = 0
    neg_samples = 0
    for i in range(num_samples):
        if data[i, len(data[i]) - 1] == 1:
            pos_samples += 1
    neg_samples = num_samples - pos_samples
    if pos_samples == 0 or neg_samples == 0:
        return 0
    entropy = -(pos_samples/num_samples) * math.log(pos_samples/num_samples, 2) - (neg_samples/num_samples) * math.log(neg_samples/num_samples, 2)
    return entropy


def most_common_label(data):
    num_samples = len(data)
    pos_samples = 0
    neg_samples = 0
    for i in range(num_samples):
        if data[i, len(data[0]) - 1] == 1:
            pos_samples += 1
    neg_samples = num_samples - pos_samples
    if pos_samples >= neg_samples:
        return 1
    else:
        return 0


def information_gain(data, feature):
    # feature is the column index of the feature within data
    if len(np.unique(data[:, feature])) == 2:
        subset1 = []
        subset2 = []
        ratios = np.zeros(2)
        for i in range(len(data)):
            if(data[i, feature]) == 0:
                subset1.append(data[i])
                ratios[0] += 1
            else:
                subset2.append(data[i])
                ratios[1] += 1
        for i in range(len(ratios)):
            ratios[i] = ratios[i] / len(data)
        if subset1:
            subset1 = np.vstack(subset1)
        if subset2:
            subset2 = np.vstack(subset2)
        gain = entropy(data) - (ratios[0] * entropy(subset1) + ratios[1] * entropy(subset1))
        return gain
    else:
        gain = entropy(data)
        for i in range(len(np.unique(data[:, feature]))):
            subset = []
            ratio = 0
            for j in range(len(data)):
                if (data[j, feature]) == i:
                    subset.append(data[j])
                    ratio += 1
            ratio = ratio/len(data)
            if subset:
                subset = np.vstack(subset)
            gain -= (ratio*entropy(subset))
        return gain


def ID3(depth, data, features):
    # data is a 2d numpy array containing features and labels
    # features is list of feature column indices in data
    # depth is starting depth, goes to 3
    global original_data
    root = Node()
    if data[0, len(data[0]) - 1] == 1 and entropy(data) == 0:
        root.prediction = 1
        return root  # positive
    if data[0, len(data[0]) - 1] == 0 and entropy(data) == 0:
        root.prediction = 0
        return root  # negative
    if len(features) == 0:
        root.prediction = most_common_label(data)
        return root
    if depth == 3:
        root.prediction = most_common_label(data)
        return root
    else:
        highest_gain = 0
        selected_feature = 0
        for feature in features:
            gain = information_gain(data, feature)
            if gain > highest_gain:
                highest_gain = gain
                selected_feature = feature
        root.feature = selected_feature
        # loop through bins of feature
        temp_feats = features[:]
        temp_feats.remove(selected_feature)
        for i in range(len(np.unique(original_data[:, selected_feature]))):
            # subset data
            subset = []
            for row in range(len(data)):
                if data[row, selected_feature] == i + 1:
                    subset.append(data[row])
            if subset:
                subset = np.vstack(subset)
            if len(subset) == 0:
                leaf = Node()
                leaf.prediction = most_common_label(data)
                root.children.append(leaf)
            else:
                root.children.append(ID3(depth + 1, subset, temp_feats))
    return root


def predict(example, node):
    feature_index = node.feature
    if node.prediction != -1:
        prediction = node.prediction
        return prediction
    else:
        return predict(example, node.children[int(example[feature_index]) - 1])


def calc_accuracy(data, tree):
    correct_pred = 0
    for i in range(len(data)):
        prediction = predict(data[i], tree)
        if prediction == data[i, len(data[i]) - 1]:
            correct_pred += 1
    return correct_pred/len(data)


def plot_data(file, num):
    df = pd.read_csv(file, header=None)
    data = df.to_numpy()
    global original_data
    original_data = data
    features = []
    for i in range(len(data[0]) - 1):
        features.append(i)
    tree = ID3(0, data, features)
    fake_data = np.asarray([[1, 1], [1, 2], [1, 3], [1, 4],
                            [2, 1], [2, 2], [2, 3], [2, 4],
                            [3, 1], [3, 2], [3, 3], [3, 4],
                            [4, 1], [4, 2], [4, 3], [4, 4]])
    predictions = []
    for i in range(len(fake_data)):
        predictions.append(predict(fake_data[i], tree))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(data[:99, 0], data[:99, 1], s=4, c='b', marker="o")
    ax1.scatter(data[100:, 0], data[100:, 1], s=4, c='r', marker="o")
    title = 'Synthetic ' + str(num)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def graphs():
    files = ['data/synthetic-1.csv',
             'data/synthetic-2.csv',
             'data/synthetic-3.csv',
             'data/synthetic-4.csv']
    for i in range(len(files)):
        plot_data(files[i], i+1)


### This function is from https://vallentin.dev/2016/11/29/pretty-print-tree
def pprint_tree(node, prefix="", last=True):
    print(prefix, "`- " if last else "|- ", node.prediction, sep="")
    prefix += "   " if last else "|  "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        last = i == (child_count - 1)
        pprint_tree(child, prefix, last)
### I did not make this


def test_entropy():
    df = pd.read_csv('data/discrete-synthetic-1.csv', header=None)
    data = df.to_numpy()
    print(entropy(data))


def test_information_gain():
    df = pd.read_csv('data/discrete-pokemonStats.csv', header=None)
    data = df.to_numpy()
    print(information_gain(data, 0))
    print(information_gain(data, 30))


def test_ID3():
    df = pd.read_csv('data/discrete-synthetic-3.csv', header=None)
    data = df.to_numpy()
    global original_data
    original_data = data
    features = []
    for i in range(len(data[0]) - 1):
        features.append(i)
    tree = ID3(0, data, features)
    pprint_tree(tree)


def test_tree_error():
    files = ['data/discrete-synthetic-1.csv',
             'data/discrete-synthetic-2.csv',
             'data/discrete-synthetic-3.csv',
             'data/discrete-synthetic-4.csv',
             'data/discrete-pokemonStats.csv']
    for i in range(len(files)):
        df = pd.read_csv(files[i], header=None)
        data = df.to_numpy()
        global original_data
        original_data = data
        features = []
        for i in range(len(data[0]) - 1):
            features.append(i)
        tree = ID3(0, data, features)
        pprint_tree(tree)
        accuracy = "{:.2%}".format(calc_accuracy(data, tree))
        print("Accuracy of this tree is : ", accuracy)


# test_entropy()
# test_information_gain()
# test_ID3()
# test_tree_error()
# graphs()
