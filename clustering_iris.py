import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
import string
#from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
#pd.options.mode.use_inf_as_na = True
pd.set_option('mode.use_inf_as_na',True)
pd.options.mode.use_inf_as_na = True


features = ['text']
#features = ['gender']

target = ''

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

global_words = {}

def global_word_list(text):
    table = str.maketrans({key: None for key in string.punctuation})
    new_text = text.translate(table)
    words = new_text.split(" ")
    se = set(words)
    global global_words
    global_words = global_words.union(se)

def fixed_values(row):
    #print(row)
    if pd.isnull(row[' cubicinches']):
        print("Fixed")
        row[' cubicinches'] = 100
    # else:
    #     print(row[' cubicinches'])

    if pd.isnull(row[' weightlbs']):
        print("Fixed")
        row[' weightlbs'] = 2500
    # else:
    #     print(row[' weightlbs'])

    return row


def test():
    iris = read_csv_file("/Users/rasrivastava/DATA_SETS/IRIS/iris.data")
    iris.columns = ["A","B","C","D","E"]
    iris_unlabelled = iris.drop(columns=["E"],axis=1)

    kmeans = KMeans(n_clusters=3, n_init=100,random_state=0).fit(iris_unlabelled)
    # print(kmeans.labels_)
    #print(kmeans.labels_.shape)
    labels = pd.DataFrame(kmeans.labels_.reshape((kmeans.labels_.shape[0],1)),columns=['predicted_cluster'])
    #print(labels)

    #output = cars_processed.append(labels,axis=1)
    output = pd.concat([iris,labels],axis=1)
    #print(output)
    output.to_csv("test.txt",sep=",",encoding='utf-8')







test()