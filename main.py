#importing the libraries and modules for logistic regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#reading the csv file using the pandas library
dataset = pd.read_csv('IIS\Assignment 3\data\Q3_diabetes_dataset.csv')

#setting up the X and Y matrices
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting the data into train and test sets int the 70:30 proportion
train, test = train_test_split(dataset, test_size=0.3, random_state=42)
x, y = dataset.iloc[:, :-1], dataset["Outcome"]
x_train, y_train = train.iloc[:, :-1], train["Outcome"]
x_test, y_test = test.iloc[:, :-1], test["Outcome"]

#setting up the logistic regression model
clf = LogisticRegression(random_state=42,max_iter=10000).fit(x_train, y_train)

#predicting the predicted value of y using the test data
y_pred = clf.predict(x_test)

#printing the Y predicted values
print("Y pred: \n",y_pred)
print("\n")

y_true = y_test.values.tolist()

#printng the confusion matrix
matrix=confusion_matrix(y_true, y_pred)
print("PRINTING THE CONFUSION MATRIX :\n",matrix)

#visualising data
plt.scatter(x_test['Pregnancies'],y_pred)
plt.show()

plt.scatter(x_test['Glucose'],y_pred)
plt.show()

plt.scatter(x_test['BloodPressure'],y_pred)
plt.show()

plt.scatter(x_test['SkinThickness'],y_pred)
plt.show()

plt.scatter(x_test['Insulin'],y_pred)
plt.show()

plt.scatter(x_test['BMI'],y_pred)
plt.show()

plt.scatter(x_test['DiabetesPedigreeFunction'],y_pred)
plt.show()

plt.scatter(x_test['Age'],y_pred)
plt.show()
