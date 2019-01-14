import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

# features = ['grade',                     # grade of the loan
#             'sub_grade',                 # sub-grade of the loan
#             'short_emp',                 # one year or less of employment
#             'emp_length_num',            # number of years of employment
#             'home_ownership',            # home_ownership status: own, mortgage or rent
#             'dti',                       # debt to income ratio
#             'purpose',                   # the purpose of the loan
#             'term',                      # the term of the loan
#             'last_delinq_none',          # has borrower had a delinquincy
#             'last_major_derog_none',     # has borrower had 90 day or worse rating
#             'revol_util',                # percent of available credit being used
#             'total_rec_late_fee',        # total late fees received to day
#             ]
features = ['grade','sub_grade','short_emp','home_ownership','purpose','term','last_delinq_none','last_major_derog_none']
#features = ['grade','term']

target = 'safe_loans'

def read_csv_file(file_name):
    return pd.read_csv(file_name)


def update_grades(loan,unique_grades):
    #print(loan)
    grade = loan['grade']
    #print(grade)
    if pd.isnull(grade):
        for gr in unique_grades:
            loan[gr] = 0
    else:
        for gr in unique_grades:
            if gr == grade:
                loan[gr] = 1
            else:
                loan[gr] = 0
    return loan

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
    unique_cols = list(map(lambda x : col_name + "_" + x,unique_cols))
    print(unique_cols)
    cols_df = pd.get_dummies(unique_cols,drop_first=False)
    loans = pd.concat([loans,cols_df],axis=1)
    loans = loans.apply(lambda loan : update_cols(loan,col_name,unique_cols),axis=1)
    loans.drop([col_name],axis=1,inplace=True)
    #print(loans)
    return loans

def create_testable_data_sets(loans):
    X = loans.drop([target],axis=1)
    Y = loans[[target]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=786)
    return X_train, X_test, Y_train, Y_test

def run_dtree(X,Y):
    clf = tree.DecisionTreeClassifier(max_depth = 5)
    clf = clf.fit(X, Y)
    return clf

def validate(clf,X_test,Y_test):
    #print(log_model)
    predictions = clf.predict(X_test)
    print(classification_report(Y_test,predictions))

def run_ll(X_train,Y_train,_lambda) :
    c = 1/_lambda
    log_model = LogisticRegression(penalty='l2',C=c)
    log_model.fit(X_train, Y_train)
    return log_model


def gradient_boosting(X,Y):
    clf = GradientBoostingClassifier()
    clf = clf.fit(X,Y)
    return clf

def test():
    loans = read_csv_file("/Users/rasrivastava/DATA_SETS/LENDING_CLUB/lending-club-data.csv")
    loans['safe_loans'] = loans['bad_loans'].apply(lambda x : 1 if x==0 else -1)
    safe_loans = loans.loc[loans['safe_loans'] == 1]
    #print(safe_loans.shape)
    unsafe_loans = loans.loc[loans['bad_loans'] == 1]
    #print(unsafe_loans.shape)
    loans = loans[features + [target]]
    # unique_grades = loans['grade'].unique()
    # grades_df = pd.get_dummies(unique_grades,drop_first=False)
    # loans = pd.concat([loans,grades_df],axis=1)
    # loans = loans.apply(lambda loan : update_grades(loan,unique_grades),axis=1)

    loans = apply_one_hot_encoding(loans,'grade')
    loans = apply_one_hot_encoding(loans,'sub_grade')
    loans = apply_one_hot_encoding(loans,'home_ownership')
    loans = apply_one_hot_encoding(loans,'purpose')
    loans = apply_one_hot_encoding(loans,'term')
    X_train, X_test, Y_train, Y_test = create_testable_data_sets(loans)
    clf = run_dtree(X_train,Y_train)
    tree.export_graphviz(clf,out_file='tree.dot',max_depth=5,feature_names = list(X_train.columns.values))

    #print(clf)
    validate(clf,X_test,Y_test)
    print("===============")
    gclf = gradient_boosting(X_train,Y_train)
    validate(gclf,X_test,Y_test)
    #print(X_train)


test()