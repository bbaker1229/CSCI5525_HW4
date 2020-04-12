############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_classification.py
############################################

# Need to get this to work

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test_error_rate as ts

# Example of how the error_rate function works
N = 21283
test_predictions = np.random.random(N)
test_predictions = test_predictions.reshape(N, 1)
ts.error_rate(test_predictions)

N = 21283
alpha = 0.1
theta = 0.001
new_predictions = np.random.random(N)
new_predictions = new_predictions.reshape(N, 1)
print(ts.error_rate(new_predictions))
new_error = ts.error_rate(new_predictions)
old_error = 100
constant_val = np.sum(new_predictions**2)
i = 1
iter = []
error_lst = []
# for i in range(10):
while np.abs(old_error - new_error) > 0.00001:
    old_error = new_error
    old_predictions = new_predictions
    error_0 = np.apply_along_axis(ts.error_rate, 1, old_predictions)
    error_0 = error_0.reshape(-1, 1)
    error_theta = np.apply_along_axis(ts.error_rate, 1, old_predictions + theta)
    error_theta = error_theta.reshape(-1, 1)
    delta_loss = (error_theta - error_0) / theta
    new_predictions = old_predictions - alpha * delta_loss
    new_predictions = new_predictions * np.sqrt(constant_val / np.sum(new_predictions ** 2))
    print(ts.error_rate(new_predictions))
    new_error = ts.error_rate(new_predictions)
    iter.append(i)
    i += 1
    error_lst.append(new_error)
    # print(iter)
    # print(error_lst)

# Create the misclassification plot
plt.xlabel('Number of Weak Learners')
plt.ylabel('Error')
plt.title('Error Improvement based on boosting')
plt.grid()
plt.plot(error_lst)
plt.show()
