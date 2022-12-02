import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class NN:
    def __init__(self, learning_rate=0.001, max_iter=1000, bias=0, threshold="Sigmoid"):
        self.max_iter = max_iter
        # self.hidden_layers = hidden_layers
        # self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.bias = bias
        self.threshold = threshold
        self.W = []

    def fThreshold(self, x):
        if self.threshold == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.threshold == "Tanh":
            return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def dthreshold(self, x):
        if self.threshold == "Sigmoid":
            return self.fThreshold(x) * (1 - self.fThreshold(x))
        elif self.threshold == "Tanh":
            return (1 - self.fThreshold(x)) * (1 + self.fThreshold(x))

    def loss(self, y, y_hat):
        return 1 / 2 * (y - y_hat) ** 2

    # it must be a 2d matrix that have number of columns equal to number of classes
    def dloss(self, y, y_hat):
        return (y - y_hat)

    # initialize the Weights, the weights are in list, each element in it is the weight between the layers, ex: W[0] is 2d array contain the weights between the input and first layer
    def layers(self, inputSize, layerSizes, numOfOutput):
        self.W.append(np.random.rand(inputSize + self.bias, layerSizes[0]))
        for i in range(len(layerSizes) - 1):
            self.W.append(np.random.rand(layerSizes[i] + self.bias, layerSizes[i + 1]))
        self.W.append(np.random.rand(layerSizes[-1] + self.bias, numOfOutput))

    # the input is list of input + output value of every layer
    def forward(self, X):
        input = []
        net = []
        if self.bias == 1:
            input.append(np.hstack((np.ones([X.shape[0], 1]), X)))  # input + bias
        else:
            input.append(X)
        net.append(np.dot(input[0], self.W[0]))  # the net value for every neural in the first layer
        input.append(self.fThreshold(net[0]))  # the output of threshold in the first layer
        for i in range(len(self.W) - 1):
            if self.bias == 1:
                tmp = np.hstack((np.ones([input[len(input) - 1].shape[0], 1]), input[len(input) - 1]))
            else:
                tmp = input[len(input) - 1]
            net.append(np.dot(tmp, self.W[i + 1]))
            input.append(self.fThreshold(net[len(net) - 1]))

        self.input = input
        self.net = net
        return input[-1]

    def backward(self, X, y):
        if self.bias == 1:
            self.input[0] = self.input[0][:, 1:]  # to remove the 1 for bias
        dcost = (self.input[len(self.input) - 1] - y) * 2
        for i in range(len(self.W)):
            if i == 0:  # output layer derivative
                db = self.dthreshold(self.net[-1]) * dcost  # dc/db = dc/da * da/dz
                dw = np.dot(self.input[-2].T, db)  # dc/dw = dc/da * da/dz * dz/dw = dc/db * X
            else:
                if self.bias == 1:
                    db = self.dthreshold(self.net[-(1 + i)] * np.dot(db, self.W[-(i)][1:].T))
                    dw = np.dot(self.input[-(2 + i)].T, db)
                else:
                    db = self.dthreshold(self.net[-(1 + i)] * np.dot(db, self.W[-(i)].T))
                    dw = np.dot(self.input[-(2 + i)].T, db)
            if self.bias == 1:
                self.W[-(i + 1)][0] -= (db.sum() * self.learning_rate / X.shape[0])
                self.W[-(i + 1)][1:] -= dw * self.learning_rate / X.shape[0]
            else:
                self.W[-(i + 1)] -= dw * self.learning_rate / X.shape[0]
            # print( i,self.W[-(i+1)].mean())
        return

    def train(self, X, y, ):
        y = np.array(self.splitOutputToNeural(y))
        plot1 =[]
        plot2 = []
        plot3 = []
        plotting = []
        score = []
        for i in range(self.max_iter):
            self.forward(X)
            self.backward(X, y)
            score.append(1-self.test(X, y))
            print("score:",score[-1])
            # print(np.abs(self.forward(X) - y).sum())
            plotting.append(np.abs(self.forward(X) - y).sum()/150)
            t1,t2,t3 = self.graph(X,y)
            plot1.append(t1)
            plot2.append(t2)
            plot3.append(t3)
        plt.plot(np.arange(0, len(plotting)), plotting,c='black')
        plt.plot(np.arange(0, len(plot1)), plot1,c='yellow')
        plt.plot(np.arange(0, len(plot2)), plot2,c='green')
        plt.plot(np.arange(0, len(plot3)), plot3,c='blue')
        plt.plot(np.arange(0, len(score)), score,c='red')
        plt.show()
        print(self.W)

    def predict(self, X):
        y = self.forward(X)
        # print(y)
        for i in range(y.shape[0]):
            max = y[i].max()
            for j in range(y.shape[1]):
                if max == y[i, j]:
                    y[i, j] = 1
                else:
                    y[i, j] = 0
        return np.array(y)

    def test(self, X, y):
        err = 0
        y_hat = self.predict(X)
        for i in range(y.shape[0]):
            if (np.abs(y_hat[i]-y[i])).sum() != 0:
                err+=1
        return err/y.shape[0]

    def splitOutputToNeural(self, y):
        unique = np.unique(y)
        split = []
        y = np.array(y)
        for i in range(y.shape[0]):
            tmp = np.zeros(unique.shape)
            tmp[y[i]] = 1
            split.append(tmp)
        # for i in range(y.shape[0]):
        #     print(y[i] , split[i])
        return split

    def graph(self, X, y):
        class1 = 0
        class2 = 0
        class3 = 0
        y_hat = self.predict(X)
        for i in range(len(y)):
            if y[i][0] != y_hat[i][0]:
                class1+=1
            if y[i][1] != y_hat[i][1]:
                class2 += 1
            if y[i][2] != y_hat[i][2]:
                class3 += 1
        return class1/150,class2/150,class3/150


# species,bill_length_mm,bill_depth_mm,flipper_length_mm,gender,body_mass_g
df = pd.read_csv('penguins.csv', index_col=False, encoding="utf-8")
df.reset_index(drop=True, inplace=True)
df['gender'] = df['gender'].fillna('male')

# preprocssing
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]
# output values
y = df["species"]
# normalize the input X
x = df.drop(["species"], axis=1).values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_new = pd.DataFrame(x_scaled)
X = df_new

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = NN(learning_rate=0.01, max_iter=1000, bias=1, threshold="Tanh")
model.layers(inputSize=X_train.shape[1], layerSizes=[3, 4], numOfOutput=3)
model.train(X, y)
