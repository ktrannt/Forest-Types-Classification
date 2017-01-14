# Naive Bayes

from task_1 import *
from Ults import *
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import  GridSearchCV

# File name use for saving classifier
filename = "Naive_BayesClassifier.pkl"

# Get the data
data_p = DataProcessing()

def plot_transform(X_train_std):
    """ This function use to plot the range value of feature after scaling

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(data.X_train[data.y_train == l, 0], data.X_train[data.y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    for l, c, m, in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax2.scatter(X_train_std[data.y_train == l, 0], X_train_std[data.y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    ax1.set_title('Transformed NON-standardized training dataset after PCA')
    ax2.set_title('Transformed standardized training dataset after PCA')

    for ax in (ax1, ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()
    plt.tight_layout()

    plt.show()

def test_scaling_features():
    """ This function for testing cross validation score with feature scaling
        and PCA applied
    """
    #print data.normalize()
    # Get the data
    scaler = {"Un_scale": (data.X_train,data.X_test),
            "Normalize": data.normalize(),
            "Standard Scaler":data.standard_scaler(),
            "Min-Max Scaler": data.min_max_scaler(),
            "Robust Scaler": data.Robust_scaler()}


    plt.figure(figsize=(12, 9))
    bar_width = 0.35
    title = "Scaling feature"
    for index, (name, (X_train_scale,X_test_scale)) in enumerate(zip(scaler.keys(), scaler.values())):

        # Apply feature reduction with PCA
        pca = PCA()
        X_transform_pca = pca.fit_transform(X_train_scale)

       # Get classifier
        clf_std = GaussianNB()

        # Get cross validation score
        score = cross_val_score(clf_std,X_train_scale,data.y_train)
        score_pca = cross_val_score(clf_std,  X_transform_pca , data.y_train)

       # Plot the scores
        plt.bar(index, np.mean(score), bar_width, label=name)
        plt.text(index, np.mean(score), '%1.3f' % np.mean(score), ha='center', va='bottom')
        plt.bar(index + bar_width,np.mean(score_pca),bar_width,color='r',label = "PCA")
        plt.text(index + bar_width, np.mean(score_pca) + 0.01, '%1.3f' % np.mean(score_pca), ha='center', va='bottom')

    plt.xlabel("Scaler")
    plt.ylabel("Cross validation Scores")
    plt.title(" Cross Validation Scores with PCA applied")
    plt.xticks(np.arange(len(scaler.keys())) + bar_width,scaler.keys())

    # Save to image
    plt.savefig("%s.png" % title, bbox_inches="tight");
    plt.show()

def test_learning_curve(estimator, title, X, y, cv, n_jobs = 4):
    # Plot the learning curve

    plot_learning_curve(estimator, title, data_p.X_train, data_p.y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=n_jobs)

def test_pca_components():
    """
    This test use PCA and GridSearch to find the
    best n_component
    """

    # Normalize data
    X_train_scale, X_test_scale = data_p.normalize()

    # Get PCA
    pca = PCA()

    # Get Gaussian Bayes classifier
    clf = GaussianNB()

    # Create pipeline
    pipe = Pipeline(steps=[('pca', pca), ('bayes', clf)])

    # Number of features
    n_components = [10,15,20, 30, 40,50,54]

    # Run Gridseach to find the best n_components
    gs = GridSearchCV(pipe,dict(pca__n_components=n_components),cv=5)
    gs.fit(X_train_scale, data_p.y_train)

    # Plot the best n_components
    plt.axvline(gs.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.savefig("Bayes PCA(n_components).png", bbox_inches="tight");
    plt.show()

if(__name__ == "__main__"):
    #Get the data
    data_p = DataProcessing()

    # Normalize the data
    X_train_scale,X_test_scale = data_p.normalize()

    #Apply feature reduction with best n_component PCA
    pca = PCA(n_components=15)
    X_transform_pca = pca.fit_transform(data_p.X_train)
    X_test_transform_pca = pca.transform(data_p.X_test)

    # Build the Naive_Bayes classifier
    clf = GaussianNB()

    #test_scaling_features()
    #test_learning_curve()

    #plot_learning_curve(clf, "Naive Bayes", data_p.X_train, data_p.y_train, cv=10, n_jobs=4)

    # The best result n_components = 15
    #test_pca_components()


    #Fit and test on test set
    clf.fit(X_transform_pca, data_p.y_train)
    y_pred = Evaluate_accuracy(clf, X_transform_pca, data_p.y_train, X_test_transform_pca, data_p.y_test)

    # Get the confusion matrix
    cnf_matrix = metrics.confusion_matrix(data_p.y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = [1,2,3,4,5,6,7]

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix')
    plt.savefig("Bayes_Confusion_Matrix.png", bbox_inches="tight")
    plt.show()