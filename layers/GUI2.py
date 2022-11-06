from tkinter import *
import Single_layer_perceptron
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
import processingdata
# Global variables

df = pd.read_csv('penguins.csv', index_col=False,encoding="utf-8")
df.reset_index(drop=True, inplace=True)
df['gender'] = df['gender'].fillna('male')

# preprocssing
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]


base = Tk()

# Using the Geometry method to the form certain dimensions
base.geometry("500x500")

# Using title method to give the title to the window
base.title('Task 1')

list_of_classes = ['Adelie', 'Gentoo', 'Chinstrap']
list_of_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']


# Adelie = 0 Chinstrap = 1 Gentoo = 2 this is the encoding order
def penclass(c):
    if c == "Adelie":
        return 0
    elif c == "Gentoo":
        return 2
    else:
        return 1

def GetData():
    feature_1=feature1.get()
    feature_2=feature2.get()
    print("feature",[feature_1,feature_2])
    # for no duplicated col
    if feature_1 == feature_2:
        featurex = feature_1
    else:
        featurex = [feature_1,feature_2]

    print(featurex)
    class_1=class1.get()
    class_2=class2.get()
    print("class ",[class_1,class_2])
    classx = [penclass(class_1),penclass(class_2)]

    learning_str=learningRate.get()
    alpha=learningRate.getdouble(learning_str)
    print("alpha ",alpha)

    ebochs_str=num_of_ebochs.get()
    Epochs = int(num_of_ebochs.getdouble(ebochs_str))
    print("itr",int(Epochs))

    Bias=bias.get()
    print("bias",Bias)

    # plot the 3 classes and features
    plt.scatter(df[feature_1],df[feature_2],c=df.species)
    plt.show()
    # model inilization
    model = Single_layer_perceptron.SingleLayer(bias=Bias,max_iter = Epochs,alpha =alpha)
    # selected classes and features with preprocessing
    df_new = df[df["species"].isin(classx)]
    X = processingdata.feature_scaling(df_new[featurex])
    y = np.array(df_new["species"]).reshape(-1,1)
    print(X.shape,y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,  np.apply_along_axis(Single_layer_perceptron.fun2, 1, y, y.max()).reshape(-1, 1), random_state=42, shuffle=True, test_size=0.2)

    model.train(X_train,y_train)
    model.test(X_test,y_test)
    processingdata.plot_it(X[:,0],X[:,1],np.array(y),W=model.W)



# ------------------------------------------------------------------------------------------------------#
# Using 'Label0' widget to create class label and using place() method, set its position.
lbl_0 = Label(base, text="Select first class", width=20, font=("bold", 11))
lbl_0.place(x=60, y=45)



# the variable 'cv' is introduced to store the String Value, which by default is (empty) ""
class1 = StringVar()
drplist = OptionMenu(base, class1, *list_of_classes)
drplist.config(width=15)
class1.set(list_of_classes[0])
drplist.place(x=240, y=40)

# Using 'Label1' widget to create classes label and using place() method, set its position.
lbl_1 = Label(base, text="Select second feature", width=20, font=("bold", 11))
lbl_1.place(x=60, y=95)

# the variable 'cv' is introduced to store the String Value, which by default is (empty) ""
class2= StringVar()
drplist = OptionMenu(base, class2, *list_of_classes)
drplist.config(width=15)
class2.set(list_of_classes[0])
drplist.place(x=240, y=90)
# -------------------------------------------------------------------------------------------------------------#


# Using 'Label2' widget to create feature label and using place() method, set its position.
lbl_2 = Label(base, text="Select first feature", width=20, font=("bold", 11))
lbl_2.place(x=60, y=170)

# the variable 'cv' is introduced to store the String Value, which by default is (empty) ""
feature1 = StringVar()
drplist = OptionMenu(base, feature1, *list_of_features)
drplist.config(width=15)
feature1.set(list_of_features[0])
drplist.place(x=240, y=165)

# Using 'Label3' widget to create feature label and using place() method, set its position.
lbl_3 = Label(base, text="Select second feature", width=20, font=("bold", 11))
lbl_3.place(x=60, y=220)

# the variable 'cv' is introduced to store the String Value, which by default is (empty) ""
feature2 = StringVar()
drplist = OptionMenu(base, feature2, *list_of_features)
drplist.config(width=15)
feature2.set(list_of_features[0])
drplist.place(x=240, y=215)
# -------------------------------------------------------------------------------------------------------------#


# Using 'Label4' widget to create learning rate label and using place() method to set its position.
lbl_4 = Label(base, text="Enter learning rate", width=20, font=("bold", 11))
lbl_4.place(x=60, y=290)

# Using Enrty widget to make a text entry box for accepting the input string in text from user.
learningRate = Entry(base)
learningRate.place(x=240, y=290)

# -------------------------------------------------------------------------------------------------------------#

# Using 'Label5' widget to create Num of ebochs label and using place() method to set its position.
lbl_5 = Label(base, text="Number of ebochs", width=20, font=("bold", 11))
lbl_5.place(x=60, y=330)

# Using Enrty widget to make a text entry box for accepting the input string in text from user.
num_of_ebochs = Entry(base)
num_of_ebochs.place(x=240, y=330)

# ------------------------------------------------------------------------------------------------------------#

# Using 'Label6' widget to create Bias label and using place() method, set its position.
lbl_6 = Label(base, text="Bias", width=20, font=('bold', 10))
lbl_6.place(x=120, y=375)

# the new variable 'vars1' is created to store Integer Value, which by default is 0.
bias = IntVar()
# Using the Checkbutton widget to create a button and using place() method to set its position.
Checkbutton(base, text="", variable=bias).place(x=235, y=372)

# ----------------------------------------------------------------------------------------------------------#

# Using the Button widget, we get to create a button for submitting all the data that has been entered in the entry boxes of the form by the user.
#button=Button(base, text='Submit', width=20, bg="grey", fg='white',command).place(x=160, y=420)
b=Button(base, text='Submit',width=20,bg='brown',fg='white',command=GetData)
b.place(x=180,y=380)
b.pack()

# b2=Button(base, text='Submit',width=20,bg='brown',fg='white',command=plot_it)
# b.place(x=180,y=380)
# b.pack()

base.mainloop()
