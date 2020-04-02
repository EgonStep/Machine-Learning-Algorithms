import numpy as np
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def create_samples(X, y):
    np.random.seed(0)
    n_samples = len(X)
    percentage = 0.75
    order = np.random.permutation(n_samples)
    data = X[order]
    target = y[order]
    return populate_samples(data, n_samples, percentage, target)


def populate_samples(data, n_samples, percentage, target):
    test_target = target[int(percentage * n_samples):]
    test_data = data[int(percentage * n_samples):]
    train_target = target[:int(percentage * n_samples)]
    train_data = data[:int(percentage * n_samples)]
    return test_data, test_target, train_data, train_target


def fit_algorithm(train_data, train_target, x, algorithm_name, clf=None):
    if algorithm_name == 'perceptron':
        clf = Perceptron(penalty=x[0], alpha=x[1], max_iter=x[2])
    elif algorithm_name == 'naive_bayes':
        clf = GaussianNB(var_smoothing=x[0])
    elif algorithm_name == 'decision_tree':
        clf = DecisionTreeClassifier(max_leaf_nodes=x[0], criterion=x[1], splitter=x[2], min_samples_leaf=x[3])
    elif algorithm_name == 'multi_layer_perceptron':
        clf = MLPClassifier(hidden_layer_sizes=x[0], max_iter=x[1], learning_rate=x[2], activation=x[3])

    clf.fit(train_data, train_target)
    return clf


def define_exit(clf, test_data, test_target):
    prediction = clf.predict(test_data)
    score = clf.score(test_data, test_target)
    matrix = metrics.confusion_matrix(test_target, prediction)
    return matrix, score
