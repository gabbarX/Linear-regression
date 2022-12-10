 # importing libraries and modules for Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

#reading the csv file using the pandas library
data_set = pd.read_csv('IIS\Assignment 3\data\Q2_insurance_dataset.csv')

#setting up the X and Y matrices
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values

#Encoding the string values to numeric values using the label Encoder
#Setting up the encoder
label_encoder_x=LabelEncoder()
#Encoding the column at 1 index
x[:,1]=label_encoder_x.fit_transform(x[:,1])
#Encoding the column at 4 index
x[:,4]=label_encoder_x.fit_transform(x[:,4])
#Encoding the column at 5 index
x[:,5]=label_encoder_x.fit_transform(x[:,5])

#Splitting the data into train and test sets int the 80:20 proportion
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#setting up the linear regression model
model=LinearRegression()

#fitting the model to the training data
model.fit(X_train,y_train)

#predicting the test data
y_predicted = model.predict(X_test)

#calculating the mean absolute error
mae = mean_absolute_error(y_test, y_predicted)

#calculating the mean squared error, also known as the accuracy
rmse = mean_squared_error(y_test, y_predicted)

#printing the results
print("Mean ABSOLUTE ERROR: ",mae)
print("ACCURACY: ",rmse)
