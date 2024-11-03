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
$$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

It is used to prevent some of the features in dataset to get dominated by other features and the model do not even consider those features.
