############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_regression.py
############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test_score as ts

# Example of how the score function works
N = 21283
test_predictions = np.random.random(N)
test_predictions = test_predictions.reshape(N, 1)
ts.score(test_predictions)
