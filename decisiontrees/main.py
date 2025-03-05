import pandas as pd
import numpy as np
from utils import train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest

# initialize data
df = pd.read_csv("data/train.csv")

# preprocess data
df = df.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1)
df = df.dropna()

# convert male/female to binary labels
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype('int64')
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype('int64')

# Construct X_train, X_test, y_train, y_test
X = df.drop(columns=["Survived"]).to_numpy()
y = df["Survived"].to_numpy()

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)

df.head(n=40)

dt = DecisionTree()
dt.train(X_train, y_train)
y_pred = dt.predict(X_test)

rf = RandomForest()
rf.train(X_train, y_train, num_models=30)
rf_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

print('')
print('')
cm = confusion_matrix(y_test, rf_pred)
print(cm)

precision = precision_score(y_test, rf_pred)
recall = recall_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")