import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils import train_test_split
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.adaboost import AdaBoost
# initialize data
df = pd.read_csv("data/train.csv")

# preprocess data
df = df.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1)
df = df.dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype('int64')
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype('int64')

# Construct X_train, X_test, y_train, y_test
X = df.drop(columns=["Survived"]).to_numpy()
y = df["Survived"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Train Decision Tree
dt = DecisionTree(max_depth=5, min_leaf_size=10)
dt.train(X_train, y_train)
dt_pred = dt.predict(X_test)


# Analyze Decision Tree
cm = confusion_matrix(y_test, dt_pred)
precision = precision_score(y_test, dt_pred)
recall = recall_score(y_test, dt_pred)
f1 = f1_score(y_test, dt_pred)

print("Decision Tree: ")
print(cm)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}\n\n")


# Train Random Forest
rf = RandomForest()
rf.train(X_train, y_train, num_models=10)
rf_pred = rf.predict(X_test)

# Analyze Random Forest
cm = confusion_matrix(y_test, rf_pred)
precision = precision_score(y_test, rf_pred)
recall = recall_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)

print("Random Forest: ")
print(cm)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}\n\n")


# Train AdaBoost
ada = AdaBoost()
ada.train(X_train, y_train)
ada_pred = ada.predict(X_test)

# Analyze AdaBoost
cm = confusion_matrix(y_test, ada_pred)
precision = precision_score(y_test, ada_pred)
recall = recall_score(y_test, ada_pred)
f1 = f1_score(y_test, ada_pred)

print("AdaBoost: ")
print(cm)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")