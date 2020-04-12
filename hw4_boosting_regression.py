############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_boosting_regression.py
############################################

import numpy as np
import matplotlib.pyplot as plt
import test_score as ts

# Define constant parameters
N = 21283  # Number of items in array
alpha = 0.01  # gradient descent learning rate
theta = 0.01  # value used in score derivative

# Create an array of random predictions
new_predictions = np.random.random(N)
new_predictions = new_predictions.reshape(N, 1)
print(ts.score(new_predictions))  # What is the score for this prediction?

# Create variables to use for a loop
new_score = ts.score(new_predictions)  # save the current score
old_score = 100  # set the initial old score to be higher than the new score
i = 1  # set a counter
iter = []  # set a list to save the counter values
scores_lst = []  # set a list to save the scores

# Change the values iteratively until the scores change very little
while np.abs(old_score - new_score) > 0.00001:
    old_score = new_score  # Save the old score
    old_predictions = new_predictions.copy()  # Save the old predictions
    # Use the old predictions to find an array of scores for each value
    scores_0 = np.apply_along_axis(ts.score, 1, old_predictions)
    scores_0 = scores_0.reshape(-1, 1)
    # Add a small amount to each value and find the new score array for all values
    scores_theta = np.apply_along_axis(ts.score, 1, old_predictions + theta)
    scores_theta = scores_theta.reshape(-1, 1)
    # Find the gradient array
    delta_loss = (scores_theta - scores_0) / theta
    # Updated the predictions
    new_predictions = old_predictions - alpha * delta_loss
    print(ts.score(new_predictions))  # Print the new score
    # Save new values
    new_score = ts.score(new_predictions)
    iter.append(i)
    i += 1
    scores_lst.append(new_score)

# Create the misclassification plot
plt.xlabel('Number of Weak Learners')
plt.ylabel('Score')
plt.title('Score Improvement based on boosting')
plt.grid()
plt.plot(scores_lst)
plt.show()
