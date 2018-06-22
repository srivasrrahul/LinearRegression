import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
#BASIS_COL_LST = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade']

BASIS_COL_LST = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']
#BASIS_COL_LST = ['bedrooms','bathrooms']

def read_file(file_name):
    df1 = pd.read_csv(file_name,sep=',')
    #df1[['bedrooms','bathrooms','floors']] = df1[['bedrooms','bathrooms','floors']].apply(pd.to_numeric)
    return df1

def get_training_test_set(df,training_percent):
    return train_test_split(df,test_size=1.0-training_percent)


def linear_model(train_data):
    X_train = train_data[BASIS_COL_LST]
    #print(X_train)
    #X_train = preprocessing.scale(X_train)
    Y_train = train_data[['price']]
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    return reg

def linear_model_feature_lst(train_data,input_feature):
    #normalize
    #train_data[input_feature] = (train_data[input_feature]-train_data[input_feature].mean())/train_data[input_feature].std()
    #train_data[['price']] = (train_data[['price']]-train_data[['price']].mean())/train_data[['price']].std()
    #train_data = train_data[train_data['sqft_living'] < 2.0]
    #train_data = train_data[train_data['price'] < 1.5]
    X_train = train_data[input_feature]
    #print(X_train)
    #X_train = preprocessing.scale(X_train)
    Y_train = train_data[['price']]
    reg = linear_regression_manual(X_train,Y_train)
    #reg.fit(X_train,Y_train)
    return reg

def multi_linear_model_feature_lst(train_data,input_features):
    X_train = train_data[input_features]
    Y_train = train_data[['price']]
    reg = multi_linear_regression_manual(X_train,Y_train)
    return reg


def multi_linear_regression_manual(X,Y):
    X = (X-X.mean())/X.std()
    Y = (Y-Y.mean())/Y.std()
    cols = X.shape[1]
    #print("Cols " + str(cols))
    W = np.zeros((cols, 1))
    W = multi_linear_gradient_descent(W,X,Y,0.005,0.05)
    return W



def multi_linear_gradient_descent(W,X,Y,epsilon,step_size):
    rss = calculate_rss_multi(W,X,Y)
    #print("RSS is " + str(rss))
    s = X.shape[0]
    W_t = W
    while rss > epsilon:
        #print("rss is " + str(rss))
        Y_estimate = np.dot(X,W)
        diff = Y-Y_estimate
        partial = -(1.0/s)*(2.0*np.dot(X.T,diff))
        W_t -= step_size * partial
        rss = calculate_rss_multi(W_t,X,Y)

    print("Final RSS " + str(rss))
    return W_t


def calculate_rss_multi(W,X,Y):
    s = X.shape[0]
    Y_estimate = np.dot(X,W)
    diff = Y-Y_estimate
    #print("Diff " + str(diff))
    return (1.0/s)*np.sqrt(np.dot(diff.T,diff))


def calculate_rss(x,y,w_0,w_1):
    s = x.shape[0]
    #print(s)
    diff_value = 0.0
    #print(size)
    for i in range(s):
        #print(i)
        #print(y.iloc[i,0])
        #print(x.iloc[i,0])
        diff_value_new = 0.0
        diff_value_new = y.iloc[i,0]-(w_0 + w_1*x.iloc[i,0])
        diff_value = diff_value + (diff_value_new * diff_value_new)

    print("RSS is " + str(diff_value/s))
    return diff_value/s

#Gradient descent from basics
def gradient_descent(x,y,w_0,w_1,step_size,epsilon):
    s = x.shape[0]
    predicted_value = calculate_rss(x,y,w_0,w_1)
    loop_count = 0
    while predicted_value > epsilon:
        dim_1 = 0.0
        dim_2 = 0.0
        for i in range(s):
            #print(y.iloc[i,0])
            #print(x.iloc[i,0])
            diff_value = y.iloc[i,0] - (w_0 + w_1*x.iloc[i,0])
            #print(diff_value)
            dim_1 += diff_value
            dim_2 += diff_value * x.iloc[i, 0]

        w_0_updated = w_0 + (2.0*step_size*dim_1)/s
        w_1_updated = w_1 + (2.0*step_size*dim_2)/s
        #print(w_0_updated)
        #if (w_1_updated < w_1):
        #    print("reduced1")
        w_0 = w_0_updated
        w_1 = w_1_updated
        #print(w_0)
        #print(w_1)
        predicted_value = calculate_rss(x,y,w_0,w_1)
        loop_count += 1
    return w_0,w_1

def linear_regression_manual(x,y):
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()

    #print(x)
    #print(y)
    #plt.plot(x,y)
    #plt.show()
    w_0, w_1 = gradient_descent(x, y, 0.0, 1.0, 0.10, 0.55)
    #print(x)
    #print(y)
    plt.plot(x,y)
    x_r = np.random.uniform(-3.0,10.0,1000)
    y_r = w_0 + w_1*x_r
    plt.plot(x_r,y_r)
    plt.show()
    return w_0,w_1


def predict(reg,test_data):
    X_test = test_data[BASIS_COL_LST]
    #X_test = preprocessing.scale(X_test)
    Y_test = test_data[['price']]
    predicted_value = reg.predict(X_test)
    err = mean_squared_error(Y_test,predicted_value)
    r2 = r2_score(Y_test,predicted_value)
    return err,r2

def predict_feature_lst(reg,test_data,input_feature):
    X_test = test_data[input_feature]
    #X_test = preprocessing.scale(X_test)
    Y_test = test_data[['price']]
    predicted_value = reg.predict(X_test)
    err = mean_squared_error(Y_test,predicted_value)
    r2 = r2_score(Y_test,predicted_value)
    return err,r2



#sqft-lot and bedrooms seem to be correlated
#waterfront is useless
#sqft basement is useless
#yr-built is usless

def augment_data(data):
    data["bedrooms_squared"] = data.apply(lambda row : row['bedrooms']*row['bedrooms'],axis=1)
    data["bed_bath_rooms"] = data.apply(lambda row : row['bedrooms']*row['bathrooms'],axis=1)
    data["log_sqft_living"] = data.apply(lambda row : np.log(row['sqft_living']),axis=1)
    data["lat_plus_long"] = data.apply(lambda row : row['lat'] + row['long'],axis=1)

def test_data(df):
    #plt.scatter(df['bedrooms'],df['price'],label = 'bedrooms',color='red')
    #plt.scatter(df['bathrooms'],df['price'],label = 'bathrooms',color='blue')
    plt.scatter(df['long'],df['price'],label = 'long',color='green')
    #plt.scatter(df['sqft_above'],df['price'],label = 'sqft_above',color='purple')
    plt.show()

#df = read_file("/Users/rasrivastava/Downloads/home_data.csv")
df = read_file("/Users/rasrivastava/Downloads/kc_house_data.csv")
#test_data(df)
print("Starting Training")
augment_data(df)
print("Bedroom square mean " + str(df['bedrooms_squared'].mean()))
print("Bedbath mean" + str(df['bed_bath_rooms'].mean()))
print("log_sqft_living mean" + str(df['log_sqft_living'].mean()))
print("lat_plus_long mean" + str(df['lat_plus_long'].mean()))
#print(df)
train_data,test_data = get_training_test_set(df,0.80)

W1 = multi_linear_model_feature_lst(train_data,["sqft_living","bedrooms","bathrooms","lat","long"])
print(W1)

print("=============================")
W2 = multi_linear_model_feature_lst(train_data,["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms"])
print(W2)

print("=============================")
W3 = multi_linear_model_feature_lst(train_data,["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms","bedrooms_squared","log_sqft_living","lat_plus_long"])
print(W3)

#print(train_data)
#reg = linear_model(train_data)
#reg = linear_model_feature_lst(train_data,["sqft_living"])
#print(reg)
#
#print("Coefficient " + str(reg.coef_))
#print("Intercept " + str(reg.intercept_))
#print("End Training")
#err,r2 = predict_feature_lst(reg,test_data,["sqft_living"])

#print(err)
#print(r2)
