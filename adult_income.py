import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


features = ['age-group','workclass','education','educational-num','marital-status','occupation','race','gender','native-country']
#features = ['gender']

target = 'high_income_50K'

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

def run_ll(X_train,Y_train,_lambda) :
    c = 1/_lambda
    log_model = LogisticRegression(penalty='l2',C=c)
    log_model.fit(X_train, Y_train)
    return log_model


def gradient_boosting(X,Y):
    clf = GradientBoostingClassifier()
    clf = clf.fit(X,np.ravel(Y,order='C'))

    return clf

def age_group(age):
    if age < 20:
        return "0-20"
    if age < 25:
        return "20-25"
    if age < 30:
        return "25-30"
    if age < 35:
        return "30-35"
    if age < 40:
        return "35-40"
    if age < 45:
        return "40-45"
    if age < 50:
        return "45-50"
    else:
        return "50+"

def test():
    incomes = read_csv_file("/Users/rasrivastava/DATA_SETS/ADULT_INCOME/adult.csv")
    incomes['high_income_50K'] = incomes['income'].apply(lambda x : 1 if x==">50K" else 0)
    incomes['age-group'] = incomes['age'].apply(age_group)
    incomes = incomes.sample(frac=1)
    #print(incomes)

    high_income_count = incomes.groupby('high_income_50K').size()
    print(high_income_count)

    # higher_income = incomes[incomes.high_income_50K == 1]
    # lower_income = incomes[incomes.high_income_50K == 0]
    # lower_income = lower_income.sample(higher_income.shape[0])
    # higher_income = higher_income.append(lower_income)
    # incomes = higher_income
    # incomes = incomes.sample(frac=1)

    # high_income_count = incomes.groupby('high_income_50K').size()
    # print(high_income_count)




    #print(unsafe_loans.shape)
    incomes = incomes[features + [target]]

    #print(incomes)
    # unique_grades = loans['grade'].unique()
    # grades_df = pd.get_dummies(unique_grades,drop_first=False)
    # loans = pd.concat([loans,grades_df],axis=1)
    # loans = loans.apply(lambda loan : update_grades(loan,unique_grades),axis=1)

    for feature in features:
        print("Feature " + feature)
        incomes = apply_one_hot_encoding(incomes,feature)

    incomes = incomes.dropna()

    X_train, X_test, Y_train, Y_test = create_testable_data_sets(incomes)
    #print(X_train)
    #clf = run_dtree(X_train,Y_train)
    #tree.export_graphviz(clf,out_file='tree.dot',max_depth=5,feature_names = list(X_train.columns.values))

    #print(clf)
    #validate(clf,X_test,Y_test)
    # print("===============")
    gclf = gradient_boosting(X_train,Y_train)
    validate(gclf,X_test,Y_test)
    #print(X_train)


test()