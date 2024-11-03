# Mobile Usage of Users
Welcome to my project !!

This is a simple machine learning project which will classify people into 5 groups(Very low = 1, Low = 2, Moderate = 3, High = 4, Very high = 5 ) based upon their usage of phones.
The dataset contains 700 data members. All thanks to kaggle for this dataset. 

Dimensionality reduction technique : Principle Component Analysis.
Techniques used for model development : Xgboost.
For better results of model and to prevent overfitting of data I have used k-fold cross validation.

Let's see in detail.

# Importing important libraries
In this section I have imported the following three important libraries:
1. Pandas as pd
2. Numpy as np
3. Matplotlib as plt

# Importing the dataset
In this section I have imported the dataset.
The variable X has the columns neccesary for making a prediction such as "App Usage time", "Screen ON Time", "Battery Drained", "Data Used". The other columns does not affect the prediction. So there is no need to include them.
The variable y has the group in which people are classified based upon the data in variable X.
The Xgboost model expects the data in the dependent variable(y) to have the starting value as 0. So i have decremented all the values in y by 1.

# Splitting the data into training set and test set
Our data need to be split into training data and testing data before start of model building.
We train the model on the training data and test our model using the testing data to detect any bugs or errors in our model.
Usually the testing data consists of 20% of whole data.

# Feature Scalling
The features(data in x) are scalled using the standardizing technique which is given as
![Equation](data:image/png;base64,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)


It is used to prevent some of the features in dataset to get dominated by other features and the model do not even consider those features.
