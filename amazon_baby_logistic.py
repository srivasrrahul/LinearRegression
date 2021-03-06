import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer


features = ['name','review','rating']
#features = ['gender']

target = 'sentiment'

def read_csv_file(file_name):
    return pd.read_csv(file_name)



def update_cols(loan,col_name,unique_col_values):
    #print(loan)
    col_value = loan[col_name]
    #print(grade)
    if pd.isnull(col_value):
        for unique_col_value in unique_col_values:
            loan[unique_col_value] = 0
    else:
        for unique_col_value in unique_col_values:
            if unique_col_value == col_value:
                loan[unique_col_value] = 1
            else:
                loan[unique_col_value] = 0
    return loan

def apply_one_hot_encoding(loans,col_name):
    unique_cols = loans[col_name].unique()
    unique_cols = list(map(lambda x : col_name + "_" + str(x),unique_cols))
    #print(unique_cols)
    cols_df = pd.get_dummies(unique_cols,drop_first=False)
    loans = pd.concat([loans,cols_df],axis=1)
    loans = loans.apply(lambda loan : update_cols(loan,col_name,unique_cols),axis=1)
    loans.drop([col_name],axis=1,inplace=True)
    #print(loans)
    return loans

def create_testable_data_sets(loans):
    X = loans.drop([target],axis=1)
    Y = loans[[target]]
    #print(X)
    #print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=10)
    return X_train, X_test, Y_train, Y_test

def run_dtree(X,Y):
    clf = tree.DecisionTreeClassifier(max_depth = 5)
    clf = clf.fit(X, Y)
    return clf

def validate(clf,X_test,Y_test):
    #print(log_model)
    predictions = clf.predict(X_test)
    print(clf.score(X_test,Y_test))
    print(classification_report(Y_test,predictions))

def run_ll(X_train,Y_train,) :

    log_model = LogisticRegression(penalty='l2',C=1.0)
    log_model.fit(X_train, Y_train)
    return log_model


def gradient_boosting(X,Y):
    clf = GradientBoostingClassifier()
    clf = clf.fit(X,np.ravel(Y,order='C'))

    return clf



def test():
    review = read_csv_file("/Users/rasrivastava/DATA_SETS/AMAZON_BABY/amazon_baby_subset.csv")
    review = review.dropna()
    X_train, X_test, Y_train, Y_test = create_testable_data_sets(review)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    vectorizer.fit(X_train['review'])
    train_matrix = vectorizer.transform(X_train['review'])
    test_matrix = vectorizer.transform(X_test['review'])

    clf = run_ll(train_matrix,Y_train)
    validate(clf,test_matrix,Y_test)




test()