import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

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
    print(X_train)
    #X_train = preprocessing.scale(X_train)
    Y_train = train_data[['price']]
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    return reg


def predict(reg,test_data):
    X_test = test_data[BASIS_COL_LST]
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



def test_data(df):
    #plt.scatter(df['bedrooms'],df['price'],label = 'bedrooms',color='red')
    #plt.scatter(df['bathrooms'],df['price'],label = 'bathrooms',color='blue')
    plt.scatter(df['long'],df['price'],label = 'long',color='green')
    #plt.scatter(df['sqft_above'],df['price'],label = 'sqft_above',color='purple')
    plt.show()

df = read_file("/Users/rasrivastava/Downloads/home_data.csv")
#test_data(df)
print("Starting Training")
train_data,test_data = get_training_test_set(df,0.9)
print(train_data)
reg = linear_model(train_data)
print("End Training")
err,r2 = predict(reg,test_data)

print(err)
print(r2)
