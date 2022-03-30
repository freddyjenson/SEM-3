import math
from math import sqrt


def euclidean_dist(row1, row2):
    dist = 0.0
    for i in range(len(row1)-1):
        dist += (row1[i] - row2[i]) ** 2
    return math.sqrt(dist)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_dist(test_row, train_row)
        distances.append([train_row, dist])

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classif(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_val = [row[-1] for row in neighbors]
    prediction = max(set(output_val), key=output_val.count)
    return prediction


# 0 - target value
dataset = [[2.78, 2.55, 0],
           [1.26, 2.36, 0],
           [3.39, 4.40, 0],
           [1.38, 1.85, 0],
           [5.74, 3.56, 0],
           [3.53, 5.67, 1],
           [6.24, 3.67, 1],
           [2.45, 1.78, 1]]

prediction = predict_classif(dataset, dataset[0], 3)
print("Expected ", dataset[0][-1], " got ", prediction)
