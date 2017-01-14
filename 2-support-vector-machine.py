# Decision Tree
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt

from scipy.stats import  randint as sp_randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split,KFold
from time import time
from sklearn import  tree
from sklearn import metrics
from sklearn.pipeline import Pipeline
from Ults import  *
from task_1 import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

# Get the data
data_p = DataProcessing()

def report(results, n_top=3):
    """ Helper function to printout the report

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def test_scaling_features():
    """ This function for testing purpose only
        Will be removed in future
    """
    data = DataProcessing()

    scaler = {"Un_scale": (data.X_train,data.X_test),
            "Normalize": data.normalize(),
            "Standard Scaler":data.standard_scaler(),
            "Min-Max Scaler": data.min_max_scaler(),
            "Robust Scaler": data.Robust_scaler()}

    plt.figure(figsize=(12,9))
    bar_width = 0.35
    title = "Scaling feature Decision Tree"
    for index,(name,(X_train_scale,X_test_scale)) in enumerate(zip(scaler.keys(),scaler.values())):


        # Apply feature reduction with PCA
        pca = PCA()
        X_transform_pca = pca.fit_transform(X_train_scale)

        # # Apply feature reduction with LinearDiscriminantAnalysis
        # lda = LinearDiscriminantAnalysis()
        # X_transform_lda = lda.fit(X_train_scale, data.y_train).transform(X_train_scale)
        # X_test_transform_lda = lda.transform(X_test_scale)

       # Get classifier
        clf_std = tree.DecisionTreeClassifier()

        # Get cross validation score
        score = cross_val_score(clf_std,X_train_scale,data.y_train)
        score_pca = cross_val_score(clf_std,  X_transform_pca , data.y_train)
        #score_lda = cross_val_score(clf_std, X_transform_lda, data.y_train)

        print "%s"%name
        print score
        print score_pca
        #print score_lda

        # clf_std.fit(X_train_scale,data.y_train)
        # Evaluate_accuracy(clf_std,X_train_scale,X_test_scale)
        # print "PCA"
        # clf_std.fit(X_transform_pca, data.y_train)
        # Evaluate_accuracy(clf_std,X_transform_pca, X_test_transform_pca)

        # print "LDA"
        # clf_std.fit(X_transform_lda, data.y_train)
        # Evaluate_accuracy(clf_std,X_transform_lda, X_test_transform_lda)

        #Plot the scores
        plt.bar(index, np.mean(score), bar_width, label=name)
        plt.bar(index + bar_width,np.mean(score_pca),bar_width,color='r',label = "PCA")
        #plt.bar(index + bar_width + bar_width, np.mean(score_lda), bar_width, color='y', label="LDA")

    plt.xlabel("Scaler")
    plt.ylabel("Cross validation Scores")
    #plt.legend()
    plt.xticks(np.arange(len(scaler.keys())) + bar_width,scaler.keys())
    plt.savefig("%s.png" % title, bbox_inches="tight");
    plt.show()

def test_dimension_reduction():
    # Build pipeline
    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', tree.DecisionTreeClassifier())
    ])

    N_FEATURES_OPTIONS = [20, 39, 40, 50, 54]
    max_depth = [1, 2, 3]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__max_depth': max_depth
        },
        # {
        #     'reduce_dim': [SelectKBest(chi2),NMF()],
        #     'reduce_dim__k': N_FEATURES_OPTIONS,
        #     'classify__C': C_OPTIONS
        # },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    data = DataProcessing()

    grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid)
    grid.fit(data.X_train, data.y_train)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(max_depth), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Digit classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')
    plt.savefig("DT_Dimension_Reduction.png", bbox_inches="tight");
    plt.show()

def tune_hyperparameters():
    # Get the data
    data = DataProcessing()

    # Normalize the data
    X_train_scale,X_test_scale = data.normalize()

    # Apply feature reduction with PCA
    pca = PCA()
    X_transform_pca = pca.fit_transform(X_train_scale)
    X_test_transform_pca = pca.transform(X_test_scale)

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, n_jobs=-1)
    start = time()
    random_search.fit(X_transform_pca, data.y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
    start = time()
    grid_search.fit(X_transform_pca, data.y_train)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)

def test_learning_curve():

    data_p = DataProcessing()
    clf = tree.ExtraTreeClassifier()
    plot_learning_curve(clf, "Bayes", data_p.X_train, data_p.y_train, ylim=(0.1, 1.01), cv=10, n_jobs=4)

def test_criterion():
    """
     This test compare between entropy and gini
    """

    # Normalize the data
    X_train_scale, X_test_scale = data_p.normalize()
    clf_en = tree.DecisionTreeClassifier(criterion='entropy')
    scores = cross_val_score(clf_en, X_train_scale, data_p.y_train, cv=10, n_jobs=-1)
    print "Entropy"
    print ("Mean score:{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

    """ Decision Tree Classifier with gini"""
    clf_gini = tree.DecisionTreeClassifier(criterion='gini')
    scores = cross_val_score(clf_gini, X_train_scale, data_p.y_train, cv=10, n_jobs=-1)
    print  "Gini"
    print ("Mean score:{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def test_validation_curves():
    data_p = DataProcessing()
    clf = SVC()
    Cs = [0.1, 1, 10, 100, 1000]
    #print param_range
    #plot_learning_curve(clf, 'gamma', data_p.X_train, data_p.y_train, ylim=(0.1, 1.01), cv=10, n_jobs=4)

    plot_validation_curve(clf, 'C', Cs, "Validation Curve with SVM", data_p.X_train[:10000], data_p.y_train[:10000], cv=10,n_jobs=-1)

if(__name__ == "__main__"):
    #test_time_curve()
    #test_scaling_features()
    #test_criterion()
    #test_validation_curves()
    #tune_hyperparameters()

    X_train_scale, X_test_scale = data_p.normalize()

    # Apply feature reduction with PCA
    pca = PCA()
    X_transform_pca = pca.fit_transform(X_train_scale)
    X_test_transform_pca = pca.transform(X_test_scale)

    # Load the classifier
    clf = Load_Classifier("DecisionTrees")

    y_pred = Evaluate_accuracy(clf, X_transform_pca, data_p.y_train, X_test_transform_pca, data_p.y_test)

    cnf_matrix = metrics.confusion_matrix(data_p.y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = [1, 2, 3, 4, 5, 6, 7]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix DecisionTree')
    plt.savefig("DecisionTree Confusion matrix.png", bbox_inches="tight");
    plt.show()



