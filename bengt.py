""" Welcome to Bengt, your automatic wine connoisseur.
    All data was downloaded from:
    http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/
"""
import warnings
from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as sk_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {"KNN_c": KNeighborsClassifier(3),
               "SVC_s": SVC(gamma="scale"),
               "DTC_5": DecisionTreeClassifier(max_depth=5),
               "RFC_8": RandomForestClassifier(max_depth=8, n_estimators=15),
               "MLP_l": MLPClassifier(alpha=1, max_iter=1000),
               "ABC_l": AdaBoostClassifier(),
               "NB_g": GaussianNB(),
               "QDA": QuadraticDiscriminantAnalysis()
}

def load_data(data_path):
    all_data = pd.read_csv(data_path, sep=";")
    return all_data

def split_data(all_data, train_ratio=0.8):
    # Randomly splits the data according to the train_ratio.
    pass
    # Convert to numpy arrays and put them in convinient dicts.
    pass
    return train_data, test_data

def whiten(train_data, out_data=None):
    """ Center the data around zero and scale to [-1, 1]
    """
    out_data = out_data or train_data
    pass
    return out_data

def train(train_data, clfs=None):
    """ Train all the models in classifiers or clfs if it is given.
    """
    clfs = clfs or classifiers
    pass

def test(test_data, clfs=None):
    """ Test all the models in classifiers or clfs if it is given. Return a
        dict with clf name as keys and the results as values.
    """
    clfs = clfs or classifiers
    pass
    return results

def test_score(true_y, pred_y):
    """ Return the averaged f1-score.
    """
    precision, recall, f1, support = sk_score(true_y, pred_y, average="micro")
    return f1

def test_all_once(all_data):
    """ Test all the models in the global classifiers-dict.
    """
    # Randomly divide the data into a training and testing set.
    train_data, test_data = split_data(all_data)
    # Whithen the train data around zero and scale between -1 and 1
    train_data = whiten(train_data)
    # Train all models at once
    train(train_data)
    # Whithen the test data but with the values from the train data
    test_data = whiten(train_data, test_data)
    # Make predictions and evaluate to get a f1 score
    test_results = test(test_data)

    # Print results sorted by test score
    tr_inv = [(score, clf) for clf, score in test_results.items()]
    tr_inv.sort(reverse=True)
    for i, (score, clf) in enumerate(tr_inv):
        print(f"#{i+1}\t{clf}\t{score:.2f}")



if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    np.set_printoptions(precision=4, suppress=True)
    # Load and split the data
    data_path = join("data", "winequality-red.csv")
    all_data = load_data(data_path)

    test_all_once(all_data)
