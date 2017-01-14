import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score,cross_val_predict
from scipy.stats import sem
from sklearn import metrics
from sklearn.model_selection import learning_curve,validation_curve


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Parameters
    ----------
    cm: confusion matrix object
    title: string
            Title of confusion matrix
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print (cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Evaluate_accuracy(clf, X_train, y_train, X_test,y_test):
    """ This function use to print out the accuracy score of classifier

       Parameters
       ----------
       clf : object type that implements the "fit" and "predict" methods
               An object of that type which is cloned for each validation.
       X_train : array-like, shape (n_samples, n_features)
               Training vector, where n_samples is the number of samples and
               n_features is the number of features.

       X_test : array-like, shape (n_samples, n_features)
               Testing vector, where n_samples is the number of samples and
               n_features is the number of features.

        y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for predict;

        y_test : array-like, shape (n_samples) or (n_samples, n_features),
            Target relative to X for testing;
       """
    pred_train = cross_val_predict(clf, X_train, y_train,cv=10)
    print('\nPrediction accuracy for the training dataset')
    print('{:.2%}'.format(metrics.accuracy_score(y_train, pred_train)))
    print "Classification report"
    print metrics.classification_report(y, pred_train), "\n"
    print "Confussion matrix"
    print metrics.confusion_matrix(y, pred_train), "\n"

    pred_test = cross_val_predict(clf, X_test, y_test,cv=10)

    print('\nPrediction accuracy for the test dataset')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
    print "Classification report"
    print metrics.classification_report(y, pred_test), "\n"
    print "Confussion matrix"
    print metrics.confusion_matrix(y, pred_test), "\n"

    return pred_test

# K-fold cross-validation
def evaluate_cross_validation(clf, X, y):

    # Create a k-fold cross validation iterator
    # Evaluate on training data and mean score
    # evaluate_cross_validation(svc_1,X_train,y_train,5)
    #cv = StratifiedKFold(n_splits = 10,shuffle=True,random_state=0)

    # print ('{}{:^61} {}'.format('Interation','Training set observations', 'Testing set observations'))
    # for interation,data in enumerate(clf,start=1):
    #     print ('{:^9}{}{:^25}'.format(interation,data[0],data[1]))


    scores = cross_val_score(clf,X,y,cv=10,n_jobs=-1)
    print  scores
    print ("Mean score:{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def plot_cross_validation(predicted,y):
    #predicted = cross_val_predict(clf, X, y, cv=10)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("%s Learning Curve.png" %title, bbox_inches="tight")
    return plt

def plot_validation_curve(estimator, param_name,param_range,title, X, y, ylim=None, cv=None,
                        n_jobs=1):
    """
        Generate a simple plot of validation curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("%s"%param_name)
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name="%s"%param_name,param_range=param_range,cv=cv, n_jobs=n_jobs, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("%s Learning Curve.png" % title, bbox_inches="tight")
    plt.show()
    return plt


def report(results, n_top=3):
    """ Helper function to printout the report
        Parameters
        ----------
        results: dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
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




