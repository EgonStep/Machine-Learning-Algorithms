import time
from sklearn.datasets import load_digits
from digits.methods import create_samples, fit_algorithm, define_exit


def decision_tree_algorithm(array_test):
    start = time.time()
    x, y = load_digits(return_X_y=True)
    test_data, test_target, train_data, train_target = create_samples(x, y)

    best_score = best_matrix = best_params = -1

    for x in array_test:
        clf = fit_algorithm(train_data, train_target, x, 'decision_tree')
        matrix, score = define_exit(clf, test_data, test_target)

        if score > best_score:
            best_score, best_matrix, best_params = score, matrix, \
                                                   "vmax_leaf_nodes=%s, criterion=%s, splitter=%s, " \
                                                   "min_samples_leaf=%d" % (x[0], x[1], x[2], x[3])

    end = time.time()
    print('Decision Tree finished in ', end - start)

    return best_score, best_matrix, best_params
