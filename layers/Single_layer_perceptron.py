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

def fun2(x,t):
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
    def __init__(self, alpha=0.01,max_iter=1000,reg_constant = 0.001,bias= True):
        self.alpha = alpha
        self.W = None
        self.reg_constant = reg_constant
        self.bias = bias
        self.max_iter = max_iter
    def threshold(self,x):
        return np.apply_along_axis(fun, 1, x).reshape(-1,1)

    def loss(self, y, y_hat):
        return np.array(1 / 2 * (y - y_hat) ** 2)

    def dloss(self, y, y_hat):
        return (y - y_hat)
    def predict(self,X):
        return np.dot(X,self.W)
    def differnce(self,y,y_hat):
        fail = 0
        for i in range(y.shape[0]):
            if y[i] != y_hat[i]:
                fail+=1
        return  1 - np.abs(fail/y.shape[0])
    def train(self,X,y):
        # initialization of weights
        if self.bias:
            X = np.hstack((np.ones([X.shape[0], 1]), X))
            num = 3
        else:
            X = np.array(X)
            num = 2
        # np.random.seed(1234)
        self.W = (np.random.rand(num)).reshape(-1,1)
        if self.bias:
            self.W[0] = 1
        for i in range(self.max_iter):
            Z = np.dot(X,self.W)
            A = self.threshold(Z)
            error = self.dloss(A,y)
            self.W = self.W - self.alpha * np.dot(X.T,error) / X.shape[0]
            if(self.differnce(A,y) > 0.95):
                print("number of iterations : ",i)
                break
        print("train score", self.differnce(self.threshold(np.dot(X, self.W)),y))
        print("Weights parameters : ", self.W.reshape(1,-1))
    def test(self,X,y):
        if self.bias:
            X = np.hstack((np.ones([X.shape[0], 1]), X))
        else:
            X = np.array(X)
        Z = np.dot(X, self.W)
        A = self.threshold(Z)
        #print("testing", A.reshape(1,-1),"\n testing",y.reshape(1,-1))
        print("model test score",self.differnce(A,y))
        unique, matrix = confusion_matrix(A,y)
        print_confusion_matrix(unique, matrix)


df = pd.read_csv('penguins.csv', index_col=False,encoding="utf-8")
df.reset_index(drop=True, inplace=True)

# print(df.isna().sum())
# only 6 missing values
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]


# plt.scatter(df.bill_length_mm, df.bill_depth_mm, c=df.gender)
# # plt.show()
#
# plt.scatter(df.flipper_length_mm, df.bill_depth_mm, c=df.gender)
# # plt.show()
#
# plt.scatter(df.bill_length_mm, df.flipper_length_mm, c=df.gender)
# # plt.show()


bias = True
# if bias:
#     X = np.random.rand(n,3)
# else:
#     X = np.random.rand(n, 2)
y = np.array(df["species"]).reshape(-1,1)[50:150]
model = SingleLayer(bias=bias,max_iter = 1000,alpha =0.01)
X = feature_scaling(df[["bill_length_mm","bill_depth_mm"]])[:100]
y = np.apply_along_axis(fun2, 1, y,y.max()).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42, shuffle = True,test_size = 0.2)

