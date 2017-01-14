# Nearest Neighbour
from sklearn.neighbors import  KNeighborsClassifier
from time import  time
from Ults import *
from task_1 import *
from sklearn.model_selection import  GridSearchCV

#Get the data
data_p = DataProcessing()

def tune_parameter():
    # Get the transformed data
    X_train_scale, X_test_scale = data_p.standard_scaler()

    clf_ = KNeighborsClassifier()

    # Choose cross-validation iterator
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    param_grid = {'algorithm': ['ball_tree'], 'n_neighbors': range(1,11)}

    # run grid search
    grid_search = GridSearchCV(clf_, param_grid=param_grid, n_jobs=-1, verbose=1, cv=cv, scoring='accuracy')
    start = time()

    # ZGit the rid with data
    grid_search.fit(X_train_scale[:100000], data_p.y_train[:100000])
    report(grid_search.cv_results_, 15)
    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))

    Save_Classifier(grid_search.best_params_, file_name='kNN')

if(__name__ == "__main__"):
    #tune_parameter()

    #Normalize the data
    X_train_scale, X_test_scale = data_p.normalize()

    # Apply feature reduction with PCA
    pca = PCA()
    X_transform_pca = pca.fit_transform(X_train_scale)
    X_test_transform_pca = pca.transform(X_test_scale)

    # Load the classifier
    clf = Load_Classifier("kNN")

    y_pred = Evaluate_accuracy(clf, X_transform_pca, data_p.y_train, X_test_transform_pca, data_p.y_test)

    cnf_matrix = metrics.confusion_matrix(data_p.y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = [1, 2, 3, 4, 5, 6, 7]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix kNN')
    plt.savefig("kNN Confusion matrix.png", bbox_inches="tight");
    plt.show()