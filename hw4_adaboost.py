############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_adaboost.py
############################################

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

training_data = pd.read_csv("cancer_train.csv")
train_y = np.array(training_data["y"])
train_y = train_y.reshape(-1, 1)
train_x = training_data
train_x = train_x.drop("y", axis=1)
testing_data = pd.read_csv("cancer_test.csv")
test_y = np.array(testing_data["y"])
test_y = test_y.reshape(-1, 1)
test_x = testing_data
test_x = test_x.drop("y", axis=1)

decision_tree = DecisionTreeClassifier(max_depth=1)
decision_tree = decision_tree.fit(train_x, train_y)
decision_tree.predict(test_x)
