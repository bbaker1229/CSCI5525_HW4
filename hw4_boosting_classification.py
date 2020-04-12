############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_classification.py
############################################

import numpy as np
import matplotlib.pyplot as plt
import test_error_rate as ts

# Define constant parameter
N = 21283  # Number of items in array

# Create an array of random predictions
new_predictions = np.random.random(N)
new_predictions = new_predictions.reshape(N, 1)
new_predictions[new_predictions >= 0.5] = 1.0
new_predictions[new_predictions < 0.5] = -1.0
new_predictions[new_predictions == -1] = 0
print(ts.error_rate(new_predictions))  # what is the error for this prediction?

# Create variables to use for a loop
new_error = ts.error_rate(new_predictions)  # save the current error
iter = []  # Use to save the iteration number
error_lst = []  # Use to save the errors

# Loop through each value and change from a 1 to -1 (or zero) or otherwise
# and choose the value that gives the lower error.
for i in range(N):
    old_error = new_error  # save the old error
    old_predictions = new_predictions.copy()  # save the old predictions
    # change the ith value of the predictions
    new_predictions[new_predictions == 0] = -1
    new_predictions[i] = new_predictions[i] * -1
    new_predictions[new_predictions == -1] = 0
    # Compare new error rate to the old error rate
    if ts.error_rate(new_predictions) > ts.error_rate(old_predictions):
        new_predictions = old_predictions.copy()  # if old error was better keep old predictions.
    # Save new values
    new_error = ts.error_rate(new_predictions)
    print(new_error)
    iter.append(i)
    error_lst.append(new_error)

# Create the misclassification plot
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error Improvement based on boosting')
plt.grid()
plt.plot(error_lst)
plt.show()
