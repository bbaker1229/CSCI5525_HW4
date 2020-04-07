############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_classification.py
############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test_error_rate as ts

# Example of how the error_rate function works
N = 21283
test_predictions = np.random.random(N)
test_predictions = test_predictions.reshape(N, 1)
ts.error_rate(test_predictions)

# Examples of how Alpha and E works for boosting - Delete later.
np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 10

D1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
h1 = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])

D2 = D1 * np.exp(-0.42 * y * h1)

h2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0])

D3 = D2 * np.exp(-0.65 * y * h2)
