import pandas as pd
import numpy as np
from utils import train_test_split
from naive_bayes import NaiveBayes


if __name__ == "__main__":

    # construct data
    df = pd.read_csv("emails.csv")
    print(df['spam'].value_counts())
    X, y = df['text'].to_numpy(), df['spam'].to_numpy()

    # split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, split=0.7)

    # create model and train on data
    model = NaiveBayes()
    model.train(X_train, y_train)

    # predict, check accuracy
    y_pred = model.predict(X_test)

    # check results (positive = spam)
    true_neg = sum(np.logical_and(y_pred==0, y_test==0))/len(y_pred)
    true_pos = sum(np.logical_and(y_pred==1, y_test==1))/len(y_pred)
    false_neg = sum(np.logical_and(y_pred==0, y_test==1))/len(y_pred)
    false_pos = sum(np.logical_and(y_pred==1, y_test==0))/len(y_pred)

    acc = (true_pos+false_neg)/(true_pos+true_neg+false_pos+false_neg)
    ran_acc = (model.phi_y)**2 + (1-model.phi_y)**2

    print(acc)
    print(ran_acc)