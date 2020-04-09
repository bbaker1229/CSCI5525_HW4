############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_regression.py
############################################

# None of this works yet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test_score as ts

# Example of how the score function works
N = 21283
alpha = 0.01
test_predictions = np.random.random(N)
test_predictions = test_predictions.reshape(N, 1)
new_score = ts.score(test_predictions)
new_score
new_scores = np.apply_along_axis(ts.score, 1, test_predictions)
new_scores = new_scores.reshape(-1, 1)
new_scores
test_predictions = test_predictions - alpha * new_scores
# old_score = new_score - 1
# while new_score > old_score:
for j in range(22):
    old_preds = test_predictions
    old_score = new_score
    old_scores = new_scores
    # for i in range(N):
    #    test_predictions[i] = test_predictions[i] - alpha * ts.score(test_predictions[i])
    new_scores = np.apply_along_axis(ts.score, 1, test_predictions)
    new_scores = new_scores.reshape(-1, 1)
    if j == 0:
        test_predictions = test_predictions - alpha * new_scores
    else:
        test_predictions = test_predictions - alpha * (new_scores - old_scores)
    new_score = ts.score(test_predictions)
    print(new_score)

np.apply_along_axis(ts.score, 1, test_predictions)

N = 21283
alpha = 0.01
test_1 = np.random.random(N)
test_1 = test_1.reshape(N, 1)
test_2 = np.random.random(N)
test_2 = test_2.reshape(N, 1)

h = (ts.score(test_2) - ts.score(test_1)) / (test_2 - test_1)
# test_1 = test_2
for i in range(10):
    temp = test_2
    test_2 = test_1 - alpha * h
    test_2 = test_2.clip(min=0)
    test_1 = temp
    h = (ts.score(test_2) - ts.score(test_1)) / (test_2 - test_1)
    print(ts.score(test_2))