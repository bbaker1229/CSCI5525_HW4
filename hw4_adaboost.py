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


def plot_margin(data, learner_lst, alpha_lst):
    """
    Create a margin plot.
    :param data: The dataset used to create the model.
    :param learner_lst: A list of weak learners.
    :param alpha_lst: A list of alphas to use in the Adaboost model.
    :return: A margin plot.
    """
    margin = []
    for i in range(data.shape[0]):
        f_x = 0
        for j in range(len(learner_lst)):
            f_x += (alpha_lst[j] * learner_lst[j].predict(np.array(data.iloc[i]).reshape(1, -1)))
        margin.append(f_x[0])
    margin = margin / sum(alpha_lst)
    margin = train_y.reshape(-1) * margin
    margin.sort()
    cumulative = np.cumsum(margin)
    cumulative = cumulative / sum(margin)
    plt.plot(margin, cumulative)
    plt.title("Adaboost Margin Plot for " + str(len(learner_lst)) + " learners")
    plt.xlabel("Margin")
    plt.ylabel("Cumulative Distribution")
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


# Read and prepare training dataset
training_data = pd.read_csv("cancer_train.csv")
train_y = np.array(training_data["y"])
train_y = train_y.reshape(-1, 1)
train_y[train_y == 0] = -1
train_x = training_data
train_x = train_x.drop("y", axis=1)

# Read and prepare testing dataset
testing_data = pd.read_csv("cancer_test.csv")
test_y = np.array(testing_data["y"])
test_y = test_y.reshape(-1, 1)
test_y[test_y == 0] = -1
test_x = testing_data
test_x = test_x.drop("y", axis=1)

# Initialize weights and lists to store alpha_lst and learner_lst
D_train = np.full(train_y.shape, 1/train_y.shape[0]).reshape(-1)
D_train = D_train / np.sum(D_train)
D_test = np.full(test_y.shape, 1/test_y.shape[0]).reshape(-1)
D_test = D_test / np.sum(D_test)
trees = []
alphas = []
iter = []
train_error_lst = []
test_error_lst = []

# Run Adaboost algorithm using 100 weak learners
for i in range(100):
    iter.append(i)
    # Train the weak learner using the sample weights we calculated
    decision_tree = DecisionTreeClassifier(max_depth=1, criterion="gini")
    decision_tree = decision_tree.fit(train_x, train_y, sample_weight=D_train)
    # Store the learner
    trees.append(decision_tree)
    # Calculate epsilon
    epsilon = np.sum(D_train * (decision_tree.predict(train_x) != train_y.reshape(-1)))
    # Calculate alpha
    alpha = 0.5 * np.log((1 - epsilon) / (epsilon))
    print("Model " + str(i) + "; epsilon: " + str(epsilon) + "; alpha: " + str(alpha))
    # Store alpha
    alphas.append(alpha)
    # Calculate the new training weights and normalize them
    D_train = D_train * np.exp(-alpha * train_y.reshape(-1) * decision_tree.predict(train_x))
    D_train = D_train / np.sum(D_train)
    # Find the training set predictions and misclassification error
    f_x = 0
    for j in iter:
        f_x += (alphas[j] * trees[j].predict(train_x))
    y_train_preds = np.sign(f_x)
    train_error = np.mean(np.not_equal(y_train_preds, train_y.reshape(-1)))
    train_error_lst.append(train_error)
    # Find the testing set predictions and misclassification error
    f_x = 0
    for j in iter:
        f_x += (alphas[j] * trees[j].predict(test_x))
    y_test_preds = np.sign(f_x)
    test_error = np.mean(np.not_equal(y_test_preds, test_y.reshape(-1)))
    test_error_lst.append(test_error)
    print("Training error: " + str(train_error) + "; Testing error: " + str(test_error))
    if i in [24, 49, 74, 99]:
        plot_margin(train_x, trees, alphas)

# Create the misclassification plot
plt.xlabel('Number of Weak Learners')
plt.ylabel('Misclassification Error')
plt.title('Adaboost Misclassification Error on Cancer Dataset')
plt.grid()
plt.plot(train_error_lst)
plt.plot(test_error_lst)
plt.gca().legend(('Training Data', 'Testing Data'))
plt.show()
