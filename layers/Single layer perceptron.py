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

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class SingleLayer:
    def __init__(self, alpha=0.01,max_iter=1000,reg_constant = 0.001,bias= True):
        self.alpha = alpha
        self.W = None
        self.reg_constant = reg_constant
        self.bias = bias
        self.max_iter = max_iter
    def threshold(self,x):
        return 1 / (1 + np.exp(-x))

    def dthreshold(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def loss(self, y, y_hat):
        return 1 / 2 * (y - y_hat) ** 2

    def dloss(self, y, y_hat):
        return (y - y_hat)

    def train(self,X,y):
        if self.bias:
            X = np.hstack((np.ones([X.shape[0], 1]), X))
            num = 3
        else:
            num = 2
        np.random.seed(1234)
        W = (np.random.rand(num))
        if self.bias:
            W[0] = 1
        for i in range(self.max_iter):
            Z = np.dot(X,W)
            A = self.threshold(Z)
            error = self.loss(A,y)
            print(error)


df = pd.read_csv('penguins.csv', index_col=False,encoding="utf-8")
df.reset_index(drop=True, inplace=True)

print(df.isna().sum())
# only 6 missing values
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]


plt.scatter(df.bill_length_mm, df.bill_depth_mm, c=df.gender)
# plt.show()

plt.scatter(df.flipper_length_mm, df.bill_depth_mm, c=df.gender)
# plt.show()

plt.scatter(df.bill_length_mm, df.flipper_length_mm, c=df.gender)
# plt.show()


n = 100
bias = True

if bias:
    X = np.random.rand(n,3)
else:
    X = np.random.rand(n, 2)
y = np.random.rand(n)
model = SingleLayer(bias=bias,max_iter = 1,alpha = 1000)
## do: make the y take only two classes
model.train(df[["bill_length_mm","bill_depth_mm"]],df["species"],)


# bill_length_mm  bill_depth_mm  flipper_length_mm