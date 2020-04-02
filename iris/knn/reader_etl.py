# Changing the Iris class to integer
def replace_label(iris_class):
    if 'Iris-setosa' in iris_class:
        return 0
    elif 'Iris-versicolor' in iris_class:
        return 1
    elif 'Iris-virginica' in iris_class:
        return 2


# Open iris.data file
file = open("../../resource/iris.data")

# Read file and save in a matrix. Each element is a row
rows = file.readlines()
matrix = []

for row in rows:
    vector = row.split(',')
    for i in range(0, 4):
        # Convert numeric values to Float
        vector[i] = float(vector[i])
    # Change the class to an integer
    vector[4] = replace_label(vector[4])
    matrix.append(vector)

# print(matrix, '\n')
# print('Matrix Length: ', len(matrix), '\n')
# print('Matrix First Element: ', matrix[0], '\n')
# print('First Element Class: ', matrix[0][4], '\n')
