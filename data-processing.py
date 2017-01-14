from __future__ import division

import pickle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC
from  sklearn.neighbors import KNeighborsClassifier
""" Import data set"""
with open('forest_data.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    target = pickle.load(f)

""" Data statistics"""

# Shape and type
print(data.shape, data.dtype)
print(target.shape, target.dtype)


class DataProcessing():
    def __init__(self,test_size = 0.15,seed = 42):
        self.test_size = test_size
        self.seed = seed

        # Split the data set into training and testing set.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=self.test_size, random_state=self.seed)

    def shuffle_dataset(self):
        """This function shuffle the data set"""

        # Prepare a stratisfied train and test split
        train_size = 0.85
        test_size = 1 - train_size

        # For ease we merge the data and target
        input_dataset = np.column_stack([data, target])

        # Shuffle the dataset
        np.random.shuffle(input_dataset)

        stratified_split = StratifiedShuffleSplit(input_dataset[:,-1], test_size=test_size, n_splits=1)

        for train_indx, test_indx in stratified_split:
            X_train = input_dataset[train_indx,:-1]
            y_train = input_dataset[train_indx,-1]
            X_test = input_dataset[test_indx,:-1]
            y_test = input_dataset[test_indx,-1]

        return  X_train,y_train,X_test,y_test

    def check_missing_value(self):
        """ Check missing value of data set
            Return: True if no value missing
                    False otherwise
        """
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
        x = imp.fit_transform(data)
        return (data==x).all()

    def plot_bar(self, labels, value, subplot, title, xLabel, yLabel, save_png = False):

        # You typically want your plot to be ~1.33x wider than tall
        plt.figure(figsize=(12, 9))

        # Remove the plot frame lines
        ax = plt.subplot(subplot)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ensure that the axis ticks show up on the bottom and left of the plot
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Set x and y label
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        # Plot the classes distribution
        ax.bar(labels,value,facecolor = "#3F5D7D")

        # Add text on top of the bar
        for x, y in zip(labels, value):
            plt.text(x + 0.4, y + 0.05, '%s' % y, ha='center', va='bottom')

        # Save the plot to png file
        if(save_png):
            plt.savefig("%s.png"%title, bbox_inches="tight");
        plt.show()

    def plot_features(self):
        """This function use to plot the feature distribution of data set"""
        unique, counts = np.unique(target, return_counts=True)
        self.plot_bar(unique, counts, 111, 'Class Distribution', 'Classes', 'Count',True)

    def plot_value_subfigures(self):
        """ This function use to plot the value distribution of each feature"""

        """List of features
            Since Widerness_Area and Soil_Types are binary data
            We will not plot them
        """

        feats = {'Elevation':0,
                 'Aspect':1,
                 'Slope':2,
                 'Horizontal_Distance_To_Hydrology':3,
                 'Vertical_Distance_To_Hydrology':4,
                 'Horizontal_Distance_To_Roadways':5,
                 'Hillshade_9am':6,
                 'Hillshade_Noon':7,
                 'Hillshade_3pm':8,
                 'Horizontal_Distance_To_Fire_Points':9,
                 }

        # Set figures sie
        plt.figure(figsize=(20,10))

        # Plot each feature figure
        for index,(name,positon) in enumerate (feats.items()):
            # Due to wide range of feature value. We select bin = 10
            n_bin = 10;

            # Get the unique value of current column
            unique = np.unique(data[:, index])
            x = np.arange(len(unique))

            ax = plt.subplot(3, 4, index)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xticks()
            plt.yticks(range(5000, 30001, 5000))
            ax.hist(unique, bins=n_bin, facecolor="#3F5D7D", edgecolor='white')
            plt.title("%s" % name)


        plt.subplots_adjust(left =.02, right =.98,top = .95, bottom = .05)
        plt.show()

    def standard_scaler(self):
        """ Scaling feature using Standard scaler

            Return
            ----------
            X_train_scale : array-like, shape (n_samples, n_features)
                Training vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.

            X_test_scale : array-like, shape (n_samples, n_features)
                Test vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.
        """
        scaler = preprocessing.StandardScaler()
        X_train_scale = scaler.fit_transform(self.X_train)
        X_test_scale = scaler.transform(self.X_test)
        return X_train_scale, X_test_scale

    def normalize(self):
        """ Scaling feature using Normalize

            Return
            ----------
            X_train_scale : array-like, shape (n_samples, n_features)
                Training vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.

            X_test_scale : array-like, shape (n_samples, n_features)
                Test vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.
        """
        normalizer = preprocessing.Normalizer(norm='l2')
        X_train_scale = normalizer.fit_transform(self.X_train)
        X_test_scale = normalizer.transform(self.X_test)
        return X_train_scale, X_test_scale

    def min_max_scaler(self):
        """ Scaling feature using min max scaler

            Return
            ----------
            X_train_scale : array-like, shape (n_samples, n_features)
                Training vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.

            X_test_scale : array-like, shape (n_samples, n_features)
                Test vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_scale = min_max_scaler.fit_transform(self.X_train)
        X_test_scale = min_max_scaler.transform(self.X_test)
        return X_train_scale, X_test_scale

    def maxAbs_scaler(self):
        """ Scaling feature using Max_Abs_scaler

            Return
            ----------
             X_train_scale : array-like, shape (n_samples, n_features)
                Training vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.

            X_test_scale : array-like, shape (n_samples, n_features)
                Test vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.
        """
        maxAbs = preprocessing.MaxAbsScaler()
        X_train_scale = maxAbs.fit_transform(self.X_train)
        X_test_scale = maxAbs.transform(self.X_test)
        return X_train_scale, X_test_scale

    def Robust_scaler(self):
        """ Scaling feature using min max scaler

            Return
            ----------
             X_train_scale : array-like, shape (n_samples, n_features)
                Training vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.

            X_test_sclae : array-like, shape (n_samples, n_features)
                Test vector after applied scaling, where n_samples is the number of samples and
                n_features is the number of features.
        """
        rb = preprocessing.RobustScaler()
        X_train_scale = rb.fit_transform(self.X_train)
        X_test_scale = rb.transform(self.X_test)
        return X_train_scale, X_test_scale

    def PCA(self, X_train, X_test):
        """
        Apply PCA on X_train and X_test

        Parameters
        ----------
         X_train : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

         X_test : array-like, shape (n_samples, n_features)
            Test vector, where n_samples is the number of samples and
            n_features is the number of features.
        """

        pca = PCA()
        X_transform_pca = pca.fit_transform(X_train)
        X_test_transform_pca = pca.fit_transform(X_test)
        return X_transform_pca, X_test_transform_pca


def Save_Classifier(estimator, file_name):
    """ This function use to save the classifier model into file

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        filename: file_name: string
            Name of model to save
    """
    joblib.dump(estimator, filename=file_name)


def Load_Classifier(file_name):
    """ This function use to load the classifier model

        Parameters
        ----------
        file_name: string
            Name of model to load
    """
    return joblib.load(file_name)


if (__name__ == "__main__"):
    a = DataProcessing()

