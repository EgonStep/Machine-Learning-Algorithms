import time
from sklearn.datasets import load_digits
from digits.methods import create_samples, fit_algorithm, define_exit


def perceptron_algorithm(array_test):
    start = time.time()
    X, y = load_digits(return_X_y=True)
    test_data, test_target, train_data, train_target = create_samples(X, y)

    best_score = best_matrix = best_params = -1

    for x in array_test:
        clf = fit_algorithm(train_data, train_target, x, 'perceptron')
        matrix, score = define_exit(clf, test_data, test_target)

        if score > best_score:
            best_score, best_matrix, best_params = score, matrix, \
                                                   'penalty=%s, alpha=%f, max_iter=%d' % (
                                                       x[0], x[1], x[2]
                                                   )
    end = time.time()
    print('Perceptron finished in ', end - start)

    return best_score, best_matrix, best_params
