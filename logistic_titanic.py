import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv("/Users/rasrivastava/DATA_SETS/TITANIC/train.csv")
missing_values = train.isnull()
# #sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='Blues_r')
# sns.countplot(x='Survived', data=train,hue='Pclass')
# plt.show()

#train.dropna()
#plt.figure(figsize=(10.7))
# sns.boxplot(x='Pclass',y='Age',data = train)
# plt.show()

def compute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 35
        if Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(compute_age, axis=1)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
train.drop(['Sex','Embarked','Ticket','Name','PassengerId'], axis=1, inplace=True)
train.drop('Cabin',axis=1,inplace=True)
X = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q','S']]
Y = train['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=786)

logModel = LogisticRegression()
logModel.fit(X_train, Y_train)

predictions = logModel.predict(X_test)

print(classification_report(Y_test,predictions))

#print(logModel)

