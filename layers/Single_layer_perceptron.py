import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def feature_scaling(x):
    """ Standardisation """
    standardisation = preprocessing.StandardScaler()
    # Scaled feature
    x_after_standardisation = standardisation.fit_transform(x)
    return x_after_standardisation


def fun(x):
    if x < 0:
        return -1
    else:
        return 1


def fun2(x, t):
    if t == 2:
        if x <= 1:
            return -1
        else:
            return 1
    else:
        if x < 1:
            return -1
        else:
            return 1


# confusion matrix
def confusion_matrix(actual, predicted):
    act = actual.flatten()
    unique = set(actual.flatten())
    pred = predicted.flatten()
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    print(lookup)
    for i in range(len(act)):
        x = lookup[act[i]]
        y = lookup[pred[i]]
        matrix[y][x] += 1
    return unique, matrix


# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


class SingleLayer:
    def __init__(self, gradient=1, alpha=0.01, max_iter=1000, reg_constant=0.001, bias=True, threshhold=0):
        self.alpha = alpha
        self.W = None
        self.reg_constant = reg_constant
        self.bias = bias
        self.max_iter = max_iter
        self.threshhold = threshhold
        self.gradient = gradient

    def threshold(self, x):
        if self.threshhold == 0:
            return np.apply_along_axis(fun, 1, x).reshape(-1, 1)
        else:
            return np.array(x)

    def loss(self, y, y_hat):
        return np.array(1 / 2 * (y - y_hat) ** 2)

    def dloss(self, y, y_hat):
        return (y - y_hat)

    def predict(self, X):
        return np.dot(X, self.W)

    def differnce(self, y, y_hat):
        y = y.flatten()
        y_hat = y_hat.flatten()
        if self.threshhold == 0:
            fail = 0
            for i in range(y.shape[0]):
                if y[i] != y_hat[i]:
                    fail += 1
            return 1 - np.abs(fail / y.shape[0])
        else:
            res = 0
            for i in range(y.shape[0]):
                res += np.power((y[i] - y_hat[i]),2)
            res = res * 0.5 / y.shape
            return res

    def train(self, X, y,stopValue):

        # initialization of weights
        if self.bias:
            X = np.hstack((np.ones([X.shape[0], 1]), X))
            num = 3
        else:
            X = np.array(X)
            num = 2
        # np.random.seed(1234)
        self.W = (np.random.rand(num)).reshape(-1, 1)
        if self.bias:
            self.W[0] = 1

        # Updating weights loop
        for i in range(self.max_iter):
            # MSE and stochastic graident
            if self.gradient:
                for j in range(y.shape[0]):
                    Z = np.dot(X[j], self.W)
                    A = self.threshold(np.array(Z).reshape(1, 1))
                    error = self.dloss(y,A)
                    # print("gradient ", self.alpha * np.dot(X.T, error).reshape(1, -1) / X.shape[0],
                    #       " Weights before change", self.W.reshape(1, -1))
                    self.W = self.W +  np.dot(X.T, self.alpha * error/ X.shape[0])

                    # print("Weights after ", self.W.reshape(1, -1), "\n")
            # single perceptron using batch
            else:
                Z = np.dot(X, self.W)
                A = self.threshold(Z)
                error = self.dloss(y, A)
                # print("error and weigts",error.sum(), self.W.reshape(1,-1))
                # print("gradient ", self.alpha * np.dot(X.T, error).reshape(1, -1) / X.shape[0],
                #       " Weights before change", self.W.reshape(1, -1))
                self.W = self.W + (self.alpha * np.dot(X.T, error)) / X.shape[0] - np.abs(self.W.sum()) * self.alpha / \
                         X.shape[0]
                # print("Weights after ", self.W.reshape(1, -1), "\n")

            # check for stopping criteria
            Z = np.dot(X, self.W)
            A = self.threshold(Z)
            error = self.dloss(y, A)

            if self.threshhold == 0:
                if (self.differnce(A, y) > 0.97):
                    print("number of iterations : ", i)
                    break
            else:
                if (self.differnce(A, y) < stopValue and i > 50):
                    print("number of iterations MSE: ", i)
                    break

    def test(self, X, y):
        if self.bias:
            X = np.hstack((np.ones([X.shape[0], 1]), X))
        else:
            X = np.array(X)
        Z = np.dot(X, self.W)
        A = self.threshold(Z)
        print("model test score", self.differnce(A, y))
        if self.threshhold == 0:
            unique, matrix = confusion_matrix(A, y)
            print_confusion_matrix(unique, matrix)
