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
    all_data = all_data.reindex(np.random.permutation(all_data.index))
    split_idx = int(train_ratio*len(all_data.index))
    trn = all_data.iloc[:split_idx]
    tst = all_data.iloc[split_idx:]
    # Convert to numpy arrays and put them in convinient dicts.
    train_data = {"x": trn.loc[:, trn.columns != "quality"].to_numpy(),
                  "y": trn.loc[:, trn.columns == "quality"].to_numpy().reshape(-1)}
    test_data = {"x": tst.loc[:, tst.columns != "quality"].to_numpy(),
                 "y": tst.loc[:, tst.columns == "quality"].to_numpy().reshape(-1)}
    return train_data, test_data

def whiten(train_data, out_data=None):
    """ Center the data around zero and scale to [-1, 1]
    """
    out_data = out_data or train_data
    mean_val = np.mean(train_data["x"], axis=0)
    max_val = np.max(np.abs(train_data["x"]-mean_val), axis=0)
    out_data["x_w"] = out_data["x"] - mean_val
    out_data["x_w"] /= max_val
    return out_data

def train(train_data, clfs=None):
    """ Train all the models in classifiers or clfs if it is given.
    """
    clfs = clfs or classifiers
    for clf_name, clf in clfs.items():
        start_time = time()
        clf.fit(train_data["x_w"], train_data["y"])
        duration = time()-start_time

def test(test_data, clfs=None):
    """ Test all the models in classifiers or clfs if it is given. Return a
        dict with clf name as keys and the results as values.
    """
    clfs = clfs or classifiers
    votes = np.zeros((len(clfs.keys()), test_data["y"].size))
    results = {}
    for i, (clf_name, clf) in enumerate(clfs.items()):
        pred_y = clf.predict(test_data["x_w"])
        votes[i, :] = pred_y
        score = test_score(test_data["y"], pred_y)
        results[clf_name] = score
    results["votes"] = test_score(test_data["y"], stats.mode(votes)[0][0])
    return results

def test_score(true_y, pred_y):
    """ Return the averaged f1-score.
    """
    precision, recall, f1, support = sk_score(true_y, pred_y, average="micro")
    return f1

def plot_multi(test_results):
    """ Plot gaussian bellcurve for the test results given.
    """
    tr_inv = [(np.mean(scores), clf) for clf, scores in test_results.items()]
    tr_inv.sort(reverse=True)
    
    x = np.linspace(0.4, 0.8, 500)
    for mu, clf in tr_inv:
        sigma = np.std(test_results[clf])
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f"{clf}: {mu:.2f}")
    plt.grid()
    plt.legend()
    plt.show()

def plot_heatmap(heat_map, iter_x, iter_y):
    """ Draw a heat map to visualize the grid search results.
    """
    pass


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

def test_all_multi(all_data, iterations=10):
    """ Test all the models multiple times and plot the mean score and spread.
    """
    test_results = {clf: [] for clf in classifiers.keys()}
    test_results["votes"] = []
    for _ in range(iterations):
        # This part is the same as in test_all_once
        train_data, test_data = split_data(all_data)
        train_data = whiten(train_data)
        train(train_data)
        test_data = whiten(train_data, test_data)
        # Save all the results in a "big" results array
        test_results_i = test(test_data)
        for clf, score in test_results_i.items():
            test_results[clf].append(score)
    # Plot the results
    plot_multi(test_results)

def grid_search(all_data):
    """ Do a grid search to fine tune the parameters for the Random
        Forest Classifier.
    """
    search = {"est_from": 1,
              "est_to": 21,
              "est_step": 1,
              "depth_from": 1,
              "depth_to": 11,
              "depth_step":1,
              "tests_per_cell": 5}
    iter_n_estimators = list(range(search["est_from"],
                                   search["est_to"],
                                   search["est_step"]))
    iter_max_depth = list(range(search["depth_from"],
                                search["depth_to"],
                                search["depth_step"]))
    heat_map = np.zeros((len(iter_n_estimators),
                         len(iter_max_depth)))
    counter = 0
    for i, n_estimators in enumerate(iter_n_estimators):
        for j, max_depth in enumerate(iter_max_depth):
            if not counter % 10:
                prog = 100*counter/heat_map.size
                print(f"\rProgress: {prog:.2f}%", end="", flush=True)
            clf = {"rbc": RandomForestClassifier(max_depth=max_depth,
                                                 n_estimators=n_estimators)}
            test_results = []
            for _ in range(search["tests_per_cell"]):
                train_data, test_data = split_data(all_data)
                train_data = whiten(train_data)
                train(train_data, clf)
                test_data = whiten(train_data, test_data)
                test_results.append(test(test_data, clf)["rbc"])
            heat_map[i, j] = np.mean(test_results)
            counter +=1
    print()
    plot_heatmap(heat_map, iter_max_depth, iter_n_estimators)
    

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    np.set_printoptions(precision=4, suppress=True)
    # Load and split the data
    data_path = join("data", "winequality-red.csv")
    all_data = load_data(data_path)

    # test_all_once(all_data)
    # test_all_multi(all_data)
    grid_search(all_data)
