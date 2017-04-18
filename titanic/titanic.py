import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
drop_columns = ['PassengerId', 'Cabin', 'Age', 'Embarked', 'Name', 'Sex', 'Ticket', 'Fare']

print("Training data: ")
print(train_data.info())
print("Test data: ")
print(test_data.info())

def prepare_train_data():
    # Remove PassengerId column, obviously unnecessary
    x_train = train_data.drop(drop_columns, axis=1)
    x_train = x_train.drop(['Survived'], axis=1)
    y_train = train_data['Survived']

    return (x_train, y_train)

def prepare_test_data():
    x_test = test_data.drop(drop_columns,axis=1).copy()

    return x_test

def run_svm(x_train, y_train, x_test):
    svc = SVC()
    svc.fit(x_train, y_train)
    y_test = svc.predict(x_test)
    score = svc.score(x_train, y_train)

    return (y_test, score)

def write_result(y_test):
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_test
    })
    submission.to_csv('titanic.csv', index=False)


if __name__ == "__main__":
    x_train, y_train = prepare_train_data()
    x_test = prepare_test_data()
    y_test, score = run_svm(x_train, y_train, x_test)
    print("Score: ", score)
    write_result(y_test)
