from iris.knn.reader_etl import matrix
from math import sqrt
import numpy as np


def math_manhattan(test, train, k):
    confusion = np.zeros((3, 3), dtype=np.int)
    correct = 0
    mistake = 0

    for i in range(len(test)):
        distance = []
        for j in range(len(train)):
            sums = 0
            for k in range(0, 4):
                sums += abs(test[i][k] - train[j][k])
            distance.append((sums, train[j][4]))
        distance.sort()
        set_score = k_scoring(distance, k)
        if set_score == test[i][4]:
            confusion[set_score][set_score] += 1
            correct += 1
        else:
            confusion[int(test[i][4])][set_score] += 1
            mistake += 1
    return correct / (correct + mistake), confusion


def math_euclidean(test, train, k):
    confusion = np.zeros((3, 3), dtype=np.int)
    correct = 0
    mistake = 0

    for i in range(len(test)):
        distance = []
        for j in range(len(train)):
            sum = 0
            for k in range(0, 4):
                sum += (test[i][k] - train[j][k]) ** 2
            distance.append((sqrt(sum), train[j][4]))
        distance.sort()
        set_score = k_scoring(distance, k)
        if set_score == test[i][4]:
            confusion[set_score][set_score] += 1
            correct += 1
        else:
            confusion[int(test[i][4])][set_score] += 1
            mistake += 1
    return correct / (correct + mistake), confusion


def k_scoring(distance, k):
    setosa = 0
    versicolour = 0
    virginica = 0

    for j in range(0, k):
        if distance[j][1] == 0:
            setosa += 1
        elif distance[j][1] == 1:
            versicolour += 1
        elif distance[j][1] == 2:
            virginica += 1
    if setosa > versicolour and setosa > virginica:
        return 0
    elif versicolour > setosa and versicolour > virginica:
        return 1
    else:
        return 2


# matrix has all the iris values
# matrix[][4] is the class
#
# Attribute Information
# 1. sepal length in cm x
# 2. sepal width in cm y
# 3. petal length in cm z
# 4. petal width in cm w
# 5. class:
#  -- Iris Setosa
#  -- Iris Versicolour
#  -- Iris Virginica

testData = np.array(matrix[:15] + matrix[50:65] + matrix[100:115])
trainData = np.array(matrix[15:50] + matrix[65:100] + matrix[115:150])
k = 11

rates_manhattan, confusion_manhattan = math_manhattan(testData, trainData, k)
rates_euclidean, confusion_euclidean = math_euclidean(testData, trainData, k)

print('k:', k, ' - Manhattan ', rates_manhattan)
print(confusion_manhattan)
print('k:', k, ' - Euclidean ', rates_euclidean)
print(confusion_euclidean)
