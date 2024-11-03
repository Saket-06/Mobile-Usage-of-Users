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
The variable y has the group 
