import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np

def read_file(file_name):
    df = pd.read_csv(file_name,sep=',')
    df.columns = ['sepal_length','sepal_width','petal_length','petal_width','type']
    return df

def get_training_test_set(df,training_percent):
    return train_test_split(df,test_size=1.0-training_percent)


def linear_model(train_data):
    X_train = train_data[['sepal_length','sepal_width','petal_length','petal_width']]
    X_train = preprocessing.scale(X_train)
    Y_train = train_data[['type']]
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    return reg


def predict(reg,test_data):
    X_test = test_data[['sepal_length','sepal_width','petal_length','petal_width']]
    X_test = preprocessing.scale(X_test)
    Y_test = test_data[['type']]
    predicted_value = reg.predict(X_test)
    #err = mean_squared_error(Y_test,predicted_value)
    #print(err)
    Y1 = Y_test.as_matrix().reshape(1,-1)
    #print(Y_test.as_matrix().reshape(1,-1))
    Y2 = predicted_value.reshape(1,-1)
    #print(reg.score(predicted_value,Y_test))
    print(Y1)
    print(Y2)
    print(np.setdiff1d(Y1,Y2))
    #for prediction in predicted_value:



df = read_file("/Users/rasrivastava/neural_net/ML_DATA/iris.data")
train_data,test_data = get_training_test_set(df,0.5)
print(train_data)
reg = linear_model(train_data)
predict(reg,test_data)