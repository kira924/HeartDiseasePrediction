import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv("heart_disease_data.csv")

# print first 5 rows of the dataset
print(heart_data.head())

# print last 5 rows of the dataset
print(heart_data.tail())

# number of rows and columns in the dataset
print(heart_data.shape)

# getting some info about the data
print(heart_data.info())

# checking for missing values
print(heart_data.isnull().sum())

# statistical measures about the data
print(heart_data.describe())

# checking the distribution of Target Variable
print(heart_data["target"].value_counts())

# separating features and target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]
print(X)

# splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# training the LogisticRegression model with Training data
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on Training data : ", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on Test data : ", test_data_accuracy)

# making a prediction for a new data point
input_data = (20, 0, 0, 120, 233, 1, 0, 150, 0, 2.3, 0, 1, 1)


# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have a Heart Disease")
else:
    print("The Person has Heart Disease")
