import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
drop_columns = ['PassengerId', 'Cabin', 'Embarked', 'Name', 'Ticket']

# Globals
max_child_age = 19
max_mid_age = 29
min_fare = 33
max_fare = 33
best_val_score = 0

print("Training data: ")
print(train_data.info())
print("Test data: ")
print(test_data.info())

def transform_sex_column(data):
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x.lower() == "male" else 0)

    return data

def transform_fare_column(data):
    mean = data['Fare'].mean()
    std = data['Fare'].std()
    count_missing = data['Fare'].isnull().sum()
    missing_fares = np.random.randint(mean - std, mean + std, size = count_missing)
    data['Fare'][np.isnan(data['Fare'])] = missing_fares
    data['Fare'] = data['Fare'].astype(int)

    # data['Fare_Per_Person']=data['Fare']/(data['Family_Size']+1)
    # data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)
    data['Fare'] = data['Fare'].apply(lambda x: 0 if x < min_fare else 1)

    return data

def transform_family_size_column(data):
    data['Family_Size'] = data['SibSp'].astype(int) + data['Parch'].astype(int)
    # data.drop(['SibSp', 'Parch'], axis = 1)

    return data

def transform_age_column(data):
    global max_mid_age, max_child_age
    mean = data['Age'].mean()
    std = data['Age'].std()
    count_missing = data['Age'].isnull().sum()
    missing_ages = np.random.randint(mean - std, mean + std, size = count_missing)
    data['Age'][np.isnan(data['Age'])] = missing_ages
    # data['Age'] = data['Age'].astype(int)
    # data['Age_Age'] = data['Age'] * data['Age']
    # data['Age_Class']=data['Age']*data['Pclass']
    # data['Age_Sex']=data['Age']*data['Sex']
    # data['Age_Sex_Class']=data['Age']*data['Sex']*data['Pclass']
    # data['Sex_Class']=data['Sex']*data['Pclass']
    # data['Age_Family']=data['Age']*data['Family_Size']
    # data['Age_Family_Class']=data['Age']*data['Family_Size']*data['Pclass']
    # data['Age_Class'] = data['Age_Class'].astype(int)
    data['Age'] = data['Age'].apply(lambda x: 0 if x < max_child_age else 1 if x < max_mid_age else 2)

    return data

def prepare_train_data():
    x_train = train_data.drop(drop_columns, axis=1)
    x_train = transform_sex_column(x_train)
    x_train = transform_family_size_column(x_train)
    x_train = transform_age_column(x_train)
    x_train = transform_fare_column(x_train)
    y_train = x_train['Survived']
    x_train = x_train.drop(['Survived'], axis=1)

    return (x_train, y_train)

def prepare_test_data():
    x_test = test_data.drop(drop_columns,axis=1)
    x_test = transform_sex_column(x_test)
    x_test = transform_family_size_column(x_test)
    x_test = transform_age_column(x_test)
    x_test = transform_fare_column(x_test)

    return x_test

def run_svm(x_train, y_train, x_test):
    svc = SVC()
    svc.fit(x_train, y_train)
    y_test = svc.predict(x_test)
    train_score = svc.score(x_train, y_train)
    scores = cross_val_score(svc, x_train, y_train, cv=5)

    return (y_test, scores.mean(), scores.std(), train_score)

def run_random_forest(x_train, y_train, x_test):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_test = rfc.predict(x_test)
    train_score = rfc.score(x_train, y_train)
    scores = cross_val_score(rfc, x_train, y_train, cv=5)

    return (y_test, scores.mean(), scores.std(), train_score)

def write_result(y_test):
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_test
    })
    submission.to_csv('titanic.csv', index=False)

def run_experiment(n=1):
    print("Experiment : ",n)
    x_train, y_train = prepare_train_data()
    x_test = prepare_test_data()
    y_test, avg_score, std, train_score = run_svm(x_train, y_train, x_test)
    print("Training score: ", train_score)
    print("Validation Accuracy: %0.2f (+/- %0.2f)" % (avg_score, std * 2))

    return (avg_score, y_test)

def optimize_fare_feature():
    global best_val_score, max_fare, min_fare
    exp = 1
    best_min_fare = 0
    best_max_fare = 0
    for i in range(33,150):
        for j in range(150, 300):
            min_fare = i
            max_fare = j
            print("Max fare, Min fare : ", max_fare, min_fare)
            avg_score, y_test = run_experiment(exp)
            exp = exp + 1
            if best_val_score < avg_score:
                best_val_score = avg_score
                best_min_fare = min_fare
                best_max_fare = max_fare
                write_result(y_test)
    print("Best fare: ", best_min_fare, best_max_fare)
    print("Best score: ", best_val_score)

def optimize_age_feature():
    global best_val_score, max_child_age, max_mid_age
    exp = 1
    best_child_age = 0
    best_mid_age = 0
    for i in range(0,80):
        for j in range(i+1, 80):
            max_child_age = i
            max_mid_age = j
            print("Max child and Max Mid age : ", max_child_age, max_mid_age)
            avg_score, y_test = run_experiment(exp)
            exp = exp + 1
            if best_val_score < avg_score:
                best_val_score = avg_score
                best_child_age = max_child_age
                best_mid_age = max_mid_age
                write_result(y_test)
    print("Best child age: ", best_child_age)
    print("Best mid age: ", best_mid_age)
    print("Best score: ", best_val_score)

if __name__ == "__main__":
    # optimize_age_feature()
    # optimize_fare_feature()
    score, y_test = run_experiment()
    write_result(y_test)
