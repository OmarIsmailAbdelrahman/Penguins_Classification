from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


def feature_encoder(x, cols):
    if cols == 0:
        lbl = LabelEncoder()
        x = lbl.fit_transform(x)
        return 0
    lbl = LabelEncoder()
    x[cols] = lbl.fit_transform(x[cols])


def feature_scaling(x):
    """ Standardisation """
    standardisation = preprocessing.StandardScaler()
    # Scaled feature
    x_after_standardisation = standardisation.fit_transform(x)
    return x_after_standardisation

def colors(y):
    t = []
    for i in range(y.shape[0]):
        if y[i] == 0:
          t.append("red")
        elif y[i] == 1:
            t.append("blue")
        else:
            t.append("black")
    return t
def plot_it(x1,x2,y1,W):
    # Plot
    # y1 = np.apply_along_axis(colors, 1, y1)
    yy = np.array(colors(y1))
    print(x1.shape,x2.shape,y1.shape)
    plt.scatter(x1, x2, c=yy)
    if W.shape[0] == 3:
        point1 = -(x1.max()*W[1] + W[0]) / W[2]
        point2 = -(x1.min()*W[1] + W[0]) / W[2]
    else:
        point1 = -(x1.max()*W[0] ) / W[1]
        point2 = -(x1.min()*W[0]) / W[1]
    t1=[x1.min(),x1.max()]
    t2 = [point2,point1]

    plt.plot(t1,t2)
    #plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})

    # Decorate
    plt.title('Color Change')
    plt.xlabel('X1 - value')
    plt.ylabel('X2 - value')
    plt.show()