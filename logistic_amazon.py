import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def read_csv_file(file_name):
    return pd.read_csv(file_name)

def get_words(words_file):
    fd = open("/Users/rasrivastava/DATA_SETS/AMAZON_BABY/important_words.json","r")
    lines = fd.readlines()
    line = lines[0]
    line = line.replace("[","")
    line = line.replace("]","")
    line = line.replace(" ","")
    line = line.replace("\"","")
    words = line.split(",")
    return words

def update_rows(row,words):
    #print(row)
    present_words = row['review']
    #print(present_words)
    #print(row)
    if pd.isnull(present_words):
        for word in words:
            row[word] = 0
    else :
        for word in words:
            if word in present_words:
                #print("Word found")
                row[word] = 1
            else:
                #print("Word not found")
                row[word] = 0

    return row

def extract_features(words,data):
    words_df = pd.get_dummies(words,drop_first=False)
    data = pd.concat([data,words_df],axis=1)
    #print(data.keys())

    new_df = data.apply(lambda x: update_rows(x,words),axis=1)
    #print(new_df.keys())

    return new_df
    #print(new_df)
    #data.drop([''],inplace=True)


def create_testable_data_sets(data,words):
    X = data[words]
    Y = data['sentiment']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=786)
    return X_train, X_test, Y_train, Y_test


def run_ll(X_train,Y_train) :
    log_model = LogisticRegression()
    log_model.fit(X_train, Y_train)
    return log_model

def validate(log_model,X_test,Y_test):
    predictions = log_model.predict(X_test)
    print(classification_report(Y_test,predictions))



def test():
    data = read_csv_file("/Users/rasrivastava/DATA_SETS/AMAZON_BABY/amazon_baby_subset.csv")
    #print(data.info())
    #words = read_csv_file("/Users/rasrivastava/DATA_SETS/AMAZON_BABY/important_words.json")
    #print(words)
    words = get_words("/Users/rasrivastava/DATA_SETS/AMAZON_BABY/important_words.json")
    #words_df = pd.get_dummies(words)
    #print(words_df)
    data = extract_features(words,data)
    data.drop(['name','review','rating'],axis=1,inplace=True)
    X_train, X_test, Y_train, Y_test = create_testable_data_sets(data,words)
    logModel = run_ll(X_train,Y_train)
    validate(logModel,X_test,Y_test)


test()