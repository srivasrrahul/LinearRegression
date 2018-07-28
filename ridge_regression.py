import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dtype_dict = {'bathrooms':float,
              'waterfront':int,
              'sqft_above':int,
              'sqft_living15':float,
              'grade':int,
              'yr_renovated':int,
              'price':float,
              'bedrooms':float,
              'zipcode':str,
              'long':float,
              'sqft_lot15':float,
              'sqft_living':float,
              'floors':float,
              'condition':int,
              'lat':float,
              'date':str,
              'sqft_basement':int,
              'yr_built':int,
              'id':str,
              'sqft_lot':int,
              'view':int}

def read_file(file_name):
    df = pd.read_csv(file_name,sep=',',dtype = dtype_dict)
    return df

def get_training_test_set(df,training_percent):
    return train_test_split(df,test_size=1.0-training_percent)

def get_x_y(data,input_features,output_features):
    return data[input_features],data[output_features]

def augment_x(x,exp_value,feature_name):
    for i in range(exp_value):
        x["sqft_living_" + str(i+2)] = x.apply(lambda row : row[feature_name]**(i+2),axis=1)

def normalize(X,Y):
    X1 = (X - X.mean())/X.std()
    Y1 = (Y - Y.mean())/Y.std()
    return X1,Y1


def rss(W,X,Y):
    #print(W.shape)
    value = np.dot(X.T,(Y-np.dot(X,W)))
    s = X.shape[0]
    return -(1.0/s) * 2.0*(value)

#A is M*1 vector
def get_abs_value(A):

    return (np.dot(A.T,A))

def linear_regression_gd(X,Y,epsilon,step_size):
    cols = X.shape[1]
    W = np.zeros((cols, 1))
    cost = rss(W,X,Y)
    iteration = 0
    while (get_abs_value(cost)) > epsilon:
        W_updated = W - step_size*cost
        W = W_updated
        cost = rss(W,X,Y)
        print("Iteration = ",str(iteration))
        iteration = iteration + 1
        #print(W)
    return W


df = read_file("/Users/rasrivastava/DATA_SETS/kc_house_data.csv")
train_data,test_data = get_training_test_set(df,0.7)
X,Y = get_x_y(train_data,["sqft_living"],["price"])
#print(X)
augment_x(X,14,'sqft_living')
X,Y = normalize(X,Y)
#print(X)
W = linear_regression_gd(X,Y,0.001,0.05)
print(W)
#print(X.iloc[0])
