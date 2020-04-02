from digits.decision_tree.decision_tree_digits import decision_tree_algorithm
from digits.multi_layer_perceptron.mlp_digits import multi_layer_perceptron_algorithm
from digits.naive_bayes.naive_bayes_digits import gaussian_naive_bayes_algorithm
from digits.perceptron.perceptron_digits import perceptron_algorithm


# Perceptron
perceptron_test = [
    ["l2", 0.0001, 60],
    ["l1", 0.0005, 100],
    ["elasticnet", 0.001, 600]
]
perceptron_score, perceptron_matrix, perceptron_params = perceptron_algorithm(perceptron_test)

print('Best Perceptron score - ', perceptron_score)
print('Best Perceptron parameters - ', perceptron_params)
print(perceptron_matrix, '\n')

# Naive Bayes
gaussian_NB_test = [
    [1e-09], [1e-10], [1e-08]
]

gaussian_NB_score, gaussian_NB_matrix, gaussian_NB_params = gaussian_naive_bayes_algorithm(gaussian_NB_test)

print('Best Gaussian NB score - ', gaussian_NB_score)
print('Best Gaussian NB parameters - ', gaussian_NB_params)
print(gaussian_NB_matrix, '\n')

# Decision Tree
decision_tree_test = [
    [None, 'gini', 'best', 5],
    [500, 'entropy', 'random', 10],
    [50, 'gini', 'best', 4]
]

decision_tree_score, decision_tree_matrix, decision_tree_params = decision_tree_algorithm(decision_tree_test)

print('Best Decision Tree score - ', decision_tree_score)
print('Best Decision Tree parameters - ', decision_tree_params)
print(decision_tree_matrix, '\n')

# Multi Layer Perceptron
mlp_test = [
    [(100,), 150, 'constant', 'identity'],
    [(500, 50), 500, 'invscaling', 'relu'],
    [(100, 50, 20), 900, 'adaptive', 'tanh']
]

mlp_score, mlp_matrix, mlp_params = multi_layer_perceptron_algorithm(mlp_test)

print('Best MLP score - ', mlp_score)
print('Best MLP parameters - ', mlp_params)
print(mlp_matrix, '\n')
