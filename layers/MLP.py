import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVC
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Global functions
# split unique values into different col
def splitOutputToNeural(y):
    unique = np.unique(y)
    split = []
    y = np.array(y)
    for i in range(y.shape[0]):
        tmp = np.zeros(unique.shape)
        tmp[y[i]] = 1
        split.append(tmp)
    return split

# split dataframe into train and test with equal size of classes
def dataframesplit(df):
    train = np.arange(0, 30)
    train = np.append(train, np.arange(50, 80))
    train = np.append(train, np.arange(100, 130))
    test = np.arange(30.50)
    test = np.append(test, np.arange(80, 100))
    test = np.append(test, np.arange(130, 150))
    return df.iloc[train], df.iloc[test]

def confusion_matrix(y, y_hat):
    matrix = np.zeros((y.shape[1],y.shape[1]))
    for i in range(y.shape[1]):     #loop for every class
        for j in range(y.shape[0]): #loop at the data set
            if y[j,i] == 0:
                continue
            matrix[i,y_hat[j].argmax()] += 1
    return matrix



class NN:
    # model initialization
    def __init__(self, learning_rate=0.001, max_iter=1000, bias=0, threshold="Sigmoid"):
        self.max_iter = max_iter
        # self.hidden_layers = hidden_layers
        # self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.bias = bias
        self.threshold = threshold
        self.W = []

    # Threshold functions [Sigmoid, Tanh]
    def fThreshold(self, x):
        if self.threshold == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.threshold == "Tanh":
            return (1 - np.exp(-x)) / (1 + np.exp(-x))
        else:
            return x

    # derivative of the threshold functions
    def dthreshold(self, x):
        if self.threshold == "Sigmoid":
            return self.fThreshold(x) * (1 - self.fThreshold(x))
        elif self.threshold == "Tanh":
            return (1 - self.fThreshold(x)) * (1 + self.fThreshold(x))
        else:
            return x

    # cost function
    def loss(self, y, y_hat):
        return 1 / 2 * (y - y_hat) ** 2

    # derivative of the cost function
    def dloss(self, y, y_hat):
        return (y - y_hat)

    # initialize the Weights, the weights are in list, each element in it is the weight between the layers, ex: W[0] is 2d array contain the weights between the input and the next layer
    def layers(self, inputSize, layerSizes, numOfOutput):
        if len(layerSizes) != 0:
            self.W.append(np.random.normal(0,1,(inputSize + self.bias, layerSizes[0]) ))
            for i in range(len(layerSizes) - 1):
                self.W.append(np.random.normal(0,1,(layerSizes[i] + self.bias, layerSizes[i + 1])))
            self.W.append(np.random.normal(0,1,(layerSizes[-1] + self.bias, numOfOutput)))
        else:
            self.W.append(
                np.random.normal(0,1,(inputSize + self.bias, numOfOutput) ))  # condition that the is no hidden layers

    # the input is list of input + output value of every layer
    def forward(self, X):
        input = []
        net = []
        # the first layer will be calculated separately because we may need to add col of one's to the NN input for bias
        # it can be joined with the loop but for simplicity this is better
        if self.bias == 1:
            input.append(np.hstack((np.ones([X.shape[0], 1]), X)))  # NN input + bias
        else:
            input.append(X) #NN input
        net.append(np.dot(input[0], self.W[0]))  # the net value for every neural in the first layer
        input.append(self.fThreshold(net[0]))  # the output of threshold in the first layer

        # forward the rest of the network
        for i in range(len(self.W) - 1):
            # adding the bias col to the next layer input and store it in tmp, layer net = tmp*W
            if self.bias == 1:
                tmp = np.hstack((np.ones([input[len(input) - 1].shape[0], 1]), input[len(input) - 1]))
            else:
                tmp = input[len(input) - 1]
            # storing the net and the threshold of the net for the backprop
            net.append(np.dot(tmp, self.W[i + 1]))

            input.append(self.fThreshold(net[len(net) - 1]))
        # using global variable to store net and output of each layer
        self.input = input
        self.net = net
        # removing the bias col of the input for backprop as it will not be used
        if self.bias == 1:
            self.input[0] = self.input[0][:, 1:]  # to remove the 1 col in the dataset for bias
        return input[-1] # return the output layer values

    def backward(self, X, y):
        dcost = (self.input[len(self.input) - 1] - y)
        for i in range(len(self.W)):

            if i == 0:  # output layer derivative is unique because it uses the cost
                db = self.dthreshold(self.net[-1]) * dcost  # dc/db = dc/da * da/dz
                dw = np.dot(self.input[-2].T, db)  # dc/dw = dc/da * da/dz * dz/dw = dc/db * X
            else:
                # The derivation is at the Top
                if self.bias == 1:
                    db = self.dthreshold(self.net[-(1 + i)] * np.dot(db, self.W[-(i)][1:].T))
                    dw = np.dot(self.input[-(2 + i)].T, db)
                else:
                    db = self.dthreshold(self.net[-(1 + i)] * np.dot(db, self.W[-(i)].T))
                    dw = np.dot(self.input[-(2 + i)].T, db)
            if self.bias == 1: # if there is a bias update it too, else update the W
                self.W[-(i + 1)][0] -= (np.sum(db,axis=0) * self.learning_rate / X.shape[0])
                self.W[-(i + 1)][1:] -= dw * self.learning_rate / X.shape[0]
            else:
                self.W[-(i + 1)] -= dw * self.learning_rate / X.shape[0]
        return

    def train(self, X, y, ):
        # it iterates for max value and
        Y = np.array(splitOutputToNeural(y))
        # Variables for plotting
        plot1 = []
        plot2 = []
        plot3 = []
        plotting = []
        score = []
        for i in range(self.max_iter):
            self.forward(X)
            self.backward(X, Y)

            err =  np.abs(self.forward(X) - Y).sum()
            # print("error value", err)
            plotting.append(np.abs(self.forward(X) - Y).sum() / 150)
            # t1, t2, t3 = self.graphClassError(X, Y)
            # plot1.append(t1)
            # plot2.append(t2)
            # plot3.append(t3)
            if err < 80:
                print("it stopped at itr:",i)
                break
            score, matrix = self.test(X, y)
        print("train score", score)
        # plot each class error and the sum of errors for the output layer and the score
        # plt.plot(np.arange(0, len(plotting)), plotting, c='black')
        # plt.plot(np.arange(0, len(plot1)), plot1, c='yellow')
        # plt.plot(np.arange(0, len(plot2)), plot2, c='green')
        # plt.plot(np.arange(0, len(plot3)), plot3, c='blue')
        # plt.plot(np.arange(0, len(score)), score, c='red')
        # plt.show()


    # predict value using forward prop and choose the highest neural to be the output

    def test(self, X, y):
        # match the output layer with the prediction
        y = np.array(splitOutputToNeural(y))
        count = 0
        y_hat = self.forward(X)
        for i in range(y.shape[0]):
            if y_hat[i ,y[i].argmax()] == y_hat[i].max():
                count +=1
        matrix = confusion_matrix(y,y_hat)
        return count / y.shape[0],matrix

    # def graphClassError(self, X, y):
    #     # for each class see the error value for plotting
    #     class1 = 0
    #     class2 = 0
    #     class3 = 0
    #     y_hat = self.f(X)
    #     for i in range(len(y)):
    #         if y[i][0] != y_hat[i][0]:
    #             class1 += 1
    #         if y[i][1] != y_hat[i][1]:
    #             class2 += 1
    #         if y[i][2] != y_hat[i][2]:
    #             class3 += 1
    #     return class1 / 150, class2 / 150, class3 / 150


# species,bill_length_mm,bill_depth_mm,flipper_length_mm,gender,body_mass_g
# Reading data
df = pd.read_csv('penguins.csv', index_col=False, encoding="utf-8")
df.reset_index(drop=True, inplace=True)

# preprocessing
df['gender'] = df['gender'].fillna('male')
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]
train, test = dataframesplit(df)

# Data Splitting
y_train = train["species"]
y_test = test["species"]
X_train = train.drop(["species"], axis=1).values
X_test = test.drop(["species"], axis=1).values

# normalize the input X
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
# Model Training and testing

model = NN(learning_rate=0.1, max_iter=1000, bias=0, threshold="Tanh")
model.layers(inputSize=X_train.shape[1], layerSizes=[], numOfOutput=len(np.unique(y_train)))
model.train(X_train, y_train)
score,matrix =  model.test(X_test, y_test)
print("test score", score)
print("confusion matrix\n",matrix)