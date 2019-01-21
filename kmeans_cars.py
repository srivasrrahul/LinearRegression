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
    cars = read_csv_file("/Users/rasrivastava/DATA_SETS/CARS_DATA/cars.csv")
    #name_text = name_text.sample(frac=0.50)

    cars.replace('', np.nan, inplace=True)
    cars.replace(' ', np.nan, inplace=True)
    cars = cars.dropna(axis=0)
    cars.reset_index(drop=True, inplace=True)
    #cars = cars[cars[' cubinches' == ' ']]
    #print(cars.columns)
    #cars_processed = cars.drop(columns=[ ' cubicinches', ' weightlbs'],axis=1)
    cars_processed = cars.apply(fixed_values,axis=1)
    cars_processed_unlabelled = cars_processed.drop(columns=[' brand'],axis=1)
    #
    print(cars_processed.shape)
    print(cars_processed_unlabelled)
    # #print(cars.columns)
    kmeans = KMeans(n_clusters=3, n_init=40,random_state=0).fit(cars_processed_unlabelled)
    # print(kmeans.labels_)
    print(kmeans.labels_.shape)
    labels = pd.DataFrame(kmeans.labels_.reshape((kmeans.labels_.shape[0],1)),columns=['predicted_cluster'])
    print(labels)

    #output = cars_processed.append(labels,axis=1)
    output = pd.concat([cars_processed,labels],axis=1)
    print(output)


    #name_text['text'].apply(global_word_list)
    #X_train, X_test, Y_train, Y_test = create_testable_data_sets(name_text)
    # vectorizer = CountVectorizer()
    # vectorizer.fit(name_text['text'])
    # word_count = vectorizer.transform(name_text['text']).toarray()
    # print(word_count.shape)
    # cols = word_count.columns
    # print(cols)
    #word_count = word_count.sample(frac=0.1)
    # model = NearestNeighbors(metric='euclidean', algorithm='brute')
    # model.fit(word_count)
    # print(name_text[name_text['name'] == 'Barack Obama'])
    # name_text = name_text.iloc[0:0]
    # #a = word_count.shape[1]
    # print("hello")
    # a = word_count[35817].reshape((1,word_count.shape[1]))
    # print("world")
    # distances, indices = model.kneighbors(a, n_neighbors=1)
    # print(distances)
    # print(indices)
    #print(word_count.toarray())
    #test_matrix = vectorizer.transform(X_test['text'])

    #clf = run_ll(train_matrix,Y_train)
    #validate(clf,test_matrix,Y_test)




test()