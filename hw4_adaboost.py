############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_adaboost.py
############################################

# none of this works yet

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

training_data = pd.read_csv("cancer_train.csv")
train_y = np.array(training_data["y"])
train_y = train_y.reshape(-1, 1)
train_y[train_y == 0] = -1
train_x = training_data
train_x = train_x.drop("y", axis=1)
testing_data = pd.read_csv("cancer_test.csv")
test_y = np.array(testing_data["y"])
test_y = test_y.reshape(-1, 1)
test_y[test_y == 0] = -1
test_x = testing_data
test_x = test_x.drop("y", axis=1)

decision_tree = DecisionTreeClassifier(max_depth=1)
decision_tree = decision_tree.fit(train_x, train_y)
decision_tree.predict(test_x)

D_train = np.full(train_y.shape, 1/train_y.shape[0]).reshape(-1)
D_test = np.full(test_y.shape, 1/test_y.shape[0]).reshape(-1)
trees = []
trees.append(decision_tree)
alphas = []
epsilon = np.sum(D_train * (decision_tree.predict(train_x) != train_y.reshape(-1)))
print(epsilon)
for t in range(100):
    epsilon = np.sum(D_train * (decision_tree.predict((D_train * train_y.reshape(-1)).reshape(-1,1)) != train_y.reshape(-1)))
    print(epsilon)
    alpha = 0.5 * np.log((1 - epsilon) / (epsilon))
    alphas.append(alpha)
    D_train = D_train * np.exp(-alpha * train_y.reshape(-1) * decision_tree.predict(train_x))
    decision_tree = DecisionTreeClassifier(max_depth=1)
    decision_tree = decision_tree.fit((D_train * train_y.reshape(-1)).reshape(-1,1), train_y)
    decision_tree.predict((D_train * train_y.reshape(-1)).reshape(-1,1))
    trees.append(decision_tree)

    D_train = D_train * np.exp(-alpha * train_y.reshape(-1) * decision_tree.predict(train_x))



D.reshape(-1) * (trees[1].predict(train_x) != train_y.reshape(-1))
