import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class NN:

    def __init__(self, learning_rate=0.001, max_iter=1000, reg_constant=0.001, bias=0,
                 threshold="Sigmoid"):
        self.max_iter = max_iter
        # self.hidden_layers = hidden_layers
        # self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
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
    def layers(self, inputSize, numOfLayers, layerSizes, numOfOutput):
        if (len(layerSizes) != numOfLayers):
            print("Wrong number of layers")
            return
        self.W.append(np.random.rand(inputSize + self.bias, layerSizes[0]))
        for i in range(numOfLayers - 1):
            self.W.append(np.random.rand(layerSizes[i] + self.bias, layerSizes[i + 1]))
        self.W.append(np.random.rand(layerSizes[-1] + self.bias, numOfOutput))
        for i in range(len(self.W)):
            print("layer", i, "weights are", self.W[i].shape)

    # the input is list of input + output value of every layer
    def forward(self, X, y):
        input = []
        net = []

        if self.bias == 1:
            input.append(np.hstack((np.ones([X.shape[0], 1]), X)))  # input + bias
        else:
            input.append(X)
        net.append(np.dot(input[0], self.W[0]))  # the net value for every neural in the first layer
        input.append(self.fThreshold(net[0]))  # the output of threshold in the first layer
        # print(X.shape, net[0].shape, )
        # print(input[0].shape, input[1].shape)
        print("input sizes: ",0,input[0].shape) # the addition 1 in the col is because of bias
        print("input sizes: ",1,input[1].shape)
        for i in range(len(self.W) - 1):
            if self.bias == 1:
                tmp = np.hstack((np.ones([input[len(input) - 1].shape[0], 1]), input[len(input) - 1]))
            else:
                tmp = input[len(input) - 1]
            net.append(np.dot(tmp,self.W[i+1]))
            input.append(self.fThreshold(net[len(net)-1]))
            print("input sizes: ",(i+2),input[-1].shape)
        self.input = input
        self.net = net

    def backward(self, X, y):
        return

    def train(self, X, y, ):
        for i in range(1):
            self.forward(X, y)
            # self.backward(X,y)

    def predict(self, X):
        return


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

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2)

model = NN( learning_rate=1, max_iter=1000, bias=0, reg_constant=0.01,threshold="Sigmoid")
model.layers(inputSize=X_train.shape[1], numOfLayers=6, layerSizes=[3, 4,10,15,2,1], numOfOutput=3)
model.train(X_train, y_train)
