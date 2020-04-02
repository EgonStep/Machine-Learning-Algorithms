from sklearn import neural_network, datasets, metrics
import numpy as np

db = datasets.load_iris()

X = db.data
Y = db.target

np.random.seed(50)

n_samples = len(X)
partition = 0.75

order = np.random.permutation(n_samples)

X = X[order]
Y = Y[order]

X_train = X[:int(n_samples * partition)]
Y_train = Y[:int(n_samples * partition)]

X_test = X[int(n_samples * partition):]
Y_test = Y[int(n_samples * partition):]

clf = neural_network.MLPClassifier(max_iter=2000, hidden_layer_sizes=(10,5))

clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

print('Score: ', clf.score(X_test, Y_test), '\n')

matrix = metrics.confusion_matrix(Y_test, prediction)

for row in matrix:
    print(row)

# Save trained algorithm memory data
# Weight vector
print('\n', clf.coefs_)
# Accumulative Error (How much information the network`s loses for every new cycle)
print('\n', clf.loss_)
# Bias
print('\n', clf.intercepts_)
# In case the algorithm isn`t updated regularly, run a new cycle of train data, with saved weights
