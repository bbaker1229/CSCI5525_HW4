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
theta = 0.01

new_predictions = np.random.random(N)
new_predictions = new_predictions.reshape(N, 1)
print(ts.score(new_predictions))
new_score = ts.score(new_predictions)
old_score = 100
i = 1
iter = []
scores_lst = []
# for i in range(10):
while np.abs(old_score - new_score) > 0.00001:
    old_score = new_score
    old_predictions = new_predictions
    scores_0 = np.apply_along_axis(ts.score, 1, old_predictions)
    scores_0 = scores_0.reshape(-1, 1)
    scores_theta = np.apply_along_axis(ts.score, 1, old_predictions + theta)
    scores_theta = scores_theta.reshape(-1, 1)
    delta_loss = (scores_theta - scores_0) / theta
    new_predictions = old_predictions - alpha * delta_loss
    print(ts.score(new_predictions))
    new_score = ts.score(new_predictions)
    iter.append(i)
    i += 1
    scores_lst.append(new_score)
    # print(iter)
    # print(scores_lst)

# Create the misclassification plot
plt.xlabel('Number of Weak Learners')
plt.ylabel('Score')
plt.title('Score Improvement based on boosting')
plt.grid()
plt.plot(scores_lst)
plt.show()


N = 21283
alpha = 0.1
theta = 0.001
new_predictions = np.random.random(N)
new_predictions = new_predictions.reshape(N, 1)
print(ts.score(new_predictions))
new_score = ts.score(new_predictions)
old_score = 100
constant_val = np.sum(new_predictions**2)
i = 1
iter = []
scores_lst = []
# for i in range(10):
while np.abs(old_score - new_score) > 0.00001:
    old_score = new_score
    old_predictions = new_predictions
    scores_0 = np.apply_along_axis(ts.score, 1, old_predictions)
    scores_0 = scores_0.reshape(-1, 1)
    scores_theta = np.apply_along_axis(ts.score, 1, old_predictions + theta)
    scores_theta = scores_theta.reshape(-1, 1)
    delta_loss = (scores_theta - scores_0) / theta
    new_predictions = old_predictions - alpha * delta_loss
    new_predictions = new_predictions * np.sqrt(constant_val / np.sum(new_predictions ** 2))
    print(ts.score(new_predictions))
    new_score = ts.score(new_predictions)
    iter.append(i)
    i += 1
    scores_lst.append(new_score)
    # print(iter)
    # print(scores_lst)

# Create the misclassification plot
plt.xlabel('Number of Weak Learners')
plt.ylabel('Score')
plt.title('Score Improvement based on boosting')
plt.grid()
plt.plot(scores_lst)
plt.show()
