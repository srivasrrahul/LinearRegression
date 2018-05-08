import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def read_file(file_name):
    df = pd.read_csv(file_name,sep=',')
    #normalized_df = (df - df.mean())/df.std()
    #le = LabelEncoder()
    #df[["Location"]] = le.fit_transform(df[["Location"]])
    #print(df["Location"])
    #X = df["Location"]
    #df["Location"] = df["Location"].reshape(1,-1)
    #print(df["Location"])
    #one_hot_encoder = OneHotEncoder(categorical_features=["Location"])
    #df = one_hot_encoder.fit_transform (df).toarray()
    X = df["Location"].unique()
    #print(X)
    X = pd.get_dummies(X,prefix="Location")
    #print(X)
    df = pd.concat([df,X],axis=1)
    df.drop(["Location"],axis=1,inplace=True)
    #print(pd.get_dummies(df[["Location"]]))
    #df[["Location"]] = pd.get_dummies(df[["Location"]])
    print(df)
    return df

def get_training_test_set(df,training_percent):
    return train_test_split(df,test_size=1.0-training_percent)

def linear_model(train_data):
    #reg = linear_model.LinearRegression()
    enc = preprocessing.OneHotEncoder()
    #train_data["Location"] = train_data["Location"]
    #print(train_data["Location"])
    #print(train_data["Location"])
    X_train = train_data[["Bedrooms","Bathrooms","Size"]]
    X_train = preprocessing.scale(X_train)
    #X_train = train_data[["Size"]]
    #X_train = (X_train - X_train.mean())/X_train.std()
    Y_train = train_data[["Price"]]
    #Y_train = (Y_train - Y_train.mean())/Y_train.std()
    Y_train = preprocessing.scale(Y_train)
    #print(X_train)
    #print(Y_train)
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    print(reg.coef_)
    #plt.scatter(X_train,Y_train,color='black')
    #plt.show()
    return reg

def predict(reg,test_data):
    X_test = test_data[["Bedrooms","Bathrooms","Size"]]
    #X_test = (X_test - X_test.mean())/X_test.std()
    X_test = preprocessing.scale(X_test)
    Y_test = test_data[["Price"]]
    Y_test = preprocessing.scale(Y_test)
    #Y_test = (Y_test - Y_test.mean())/Y_test.std()

    predicted_value = reg.predict(X_test)
    #plt.plot(X_test,predicted_value,color='blue')
    #plt.xticks(())
    ##plt.yticks(())
    #plt.show()
    err = mean_squared_error(Y_test,predicted_value)
    print(err)

df = read_file("/Users/rasrivastava/Downloads/RealEstate.csv")
train_data,test_data = get_training_test_set(df,0.7)
print(train_data)
reg = linear_model(train_data)
predict(reg,test_data)
