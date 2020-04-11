############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw4_random_forest.py
############################################

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def build_dataset(data):
    n = data.shape[0]
    new_dataset = data.sample(n=n, replace=True)
    y = np.array(new_dataset["y"])
    y = y.reshape(-1, 1)
    y[y == 0] = -1
    x = new_dataset
    x = x.drop("y", axis=1)
    return x, y


def train_classifier(data_x, data_y, max_features):
    decision_tree = DecisionTreeClassifier(max_depth=1, criterion="gini", max_features=max_features)
    decision_tree = decision_tree.fit(data_x, data_y)
    return decision_tree


def random_forest(data, max_features, n_trees=100):
    forest = []
    for i in range(n_trees):
        train_x, train_y = build_dataset(data)
        tree = train_classifier(train_x, train_y, max_features=max_features)
        forest.append(tree)
    return forest


def fit_r_forest(model, data):
    final_data = 0
    for i in model:
        tree = i.predict(data)
        final_data += tree
    final_data /= len(model)
    final_data = np.sign(final_data)
    final_data[final_data == -1] = 0
    return final_data


training_data = pd.read_csv("health_train.csv")
testing_data = pd.read_csv("health_test.csv")

# Example of a fit and prediction of the random forest model
features = training_data.shape[1] - 1  # define the number of features (sub 1 due to target)
rf_model = random_forest(training_data, max_features=features, n_trees=100)  # Fit model
train_x = training_data  # Prepare x dataset for predictions
train_x = train_x.drop("y", axis=1)
preds = fit_r_forest(rf_model, train_x)  # Predict using X dataset
train_y = np.array(training_data["y"])  # Prepare y dataset for accuracy comparison
np.mean(np.equal(preds, train_y))  # Determine the accuracy of the model

# Vary the size of random feature sets and plot accuracy for training and test data
feature_sizes = [50, 100, 150, 200, 250]
train_accuracy = []
test_accuracy = []
for features in feature_sizes:
    rf_model = random_forest(training_data, max_features=features, n_trees=100)
    train_x = training_data
    train_x = train_x.drop("y", axis=1)
    preds = fit_r_forest(rf_model, train_x)
    train_y = np.array(training_data["y"])
    acc_val = np.mean(np.equal(preds, train_y))
    train_accuracy.append(acc_val)
    test_x = testing_data
    test_x = test_x.drop("y", axis=1)
    preds = fit_r_forest(rf_model, test_x)
    test_y = np.array(testing_data["y"])
    acc_val = np.mean(np.equal(preds, test_y))
    test_accuracy.append(acc_val)
dict = {'feature_size': feature_sizes, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
df = pd.DataFrame(dict)
df.plot(kind='line', x='feature_size', y='train_accuracy')
df.plot(kind='line', x='feature_size', y='test_accuracy', color='red', ax=plt.gca())
plt.title("Accuracy by feature_size with a random forest of 100 trees")
plt.ylabel("Accuracy")
plt.show()

# Vary the number of estimators and plot the accuracy for training and test data
tree_size = [10, 20, 40, 80, 100]
train_accuracy = []
test_accuracy = []
for n_trees in tree_size:
    rf_model = random_forest(training_data, max_features=250, n_trees=n_trees)
    train_x = training_data
    train_x = train_x.drop("y", axis=1)
    preds = fit_r_forest(rf_model, train_x)
    train_y = np.array(training_data["y"])
    acc_val = np.mean(np.equal(preds, train_y))
    train_accuracy.append(acc_val)
    test_x = testing_data
    test_x = test_x.drop("y", axis=1)
    preds = fit_r_forest(rf_model, test_x)
    test_y = np.array(testing_data["y"])
    acc_val = np.mean(np.equal(preds, test_y))
    test_accuracy.append(acc_val)
dict = {'tree_size': tree_size, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
df = pd.DataFrame(dict)
df.plot(kind='line', x='tree_size', y='train_accuracy')
df.plot(kind='line', x='tree_size', y='test_accuracy', color='red', ax=plt.gca())
plt.title("Accuracy by tree_size with a random forest of 250 max feature size")
plt.ylabel("Accuracy")
plt.show()
