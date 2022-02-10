#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

import matplotlib #collection of functions for scientific and publication-ready visualization

import numpy as np #foundational package for scientific computing

import scipy as sp #collection of functions for scientific computing and advance mathematics

import sklearn #collection of machine learning algorithms

#misc libraries
import random
import time

import functions as funk

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

data_raw = pd.read_csv('./Data/train.csv')

data_val = pd.read_csv('./Data/test.csv')

# Create a copy to play with the data
data1 = data_raw.copy(deep=True)

# Pass a reference to each of the datasets to clean both datasets at once
data_cleaner = [data1, data_val]

#funk.preview_data(data_raw)

#missing_values_data1 = funk.missing_values_table(data1)
#missing_values_data2 = funk.missing_values_table(data_val)
#print(missing_values_data1.head(30))
#print('\n', '-'*10)
#print(missing_values_data2.head(30))

#print(data1.dtypes.value_counts())
#print(data_val.dtypes.value_counts())

#print('Train columns with null values:\n', data1.isnull().sum())
#print("-"*10)

#print('Test/Validation columns with null values:\n', data_val.isnull().sum())
#print("-"*10)

#print(data_raw.describe(include = 'all'))

# CLEANING THE DATA
# ++++++++++++++++++
#data_val['Utilities'].isnull().any()
#print(data_val['Utilities'].describe)
print(len(data_val['Utilities'].unique()))
data_val['Utilities'].astype('string')
data_val = data_val.fillna('NAN')
data1 = data1.fillna('NAN')
print(data_val['Utilities'].unique())

# Encoding Categorical Values
# Create a label encoder object
le = dict()
le_count = 0

for col in data1:
    # print(col)
    if data1[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(data1[col].unique())) <= 2:
            le[col] = LabelEncoder()
            # Train on the training data
            le[col].fit(data1[col])
            funk.encode_transform(data1, col, le)

            #Keep track of how many columns were label encoded
            le_count += 1

#print(f'{le_count} coluumns were label encoded in data1.')

le = dict()
le_count = 0
# Iterate through the columns
for col in data_val:
    # print(col)
    if data_val[col].dtype == 'object':
        # IF 2 or fewer categories
        if len(list(data_val[col].astype('str').unique())) <= 2:
            le[col] = LabelEncoder()
            # Train on the testing data
            le[col].fit(data_val[col])
            funk.encode_transform(data_val, col, le)

            # Keep track of how many columns were label encoded
            le_count += 1

#print(f'{le_count} columns were label encoded in data_val')

drop_column = ['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars', 'LotFrontage']
data1.drop(drop_column, axis=1, inplace = True)
data_val.drop(drop_column, axis=1, inplace = True)

print("-"*10)
print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())

#missing_values_data1 = funk.missing_values_table(data1)
#missing_values_data2 = funk.missing_values_table(data_val)
#print(missing_values_data1.head(30))
#print('\n', '-'*10)
#print(missing_values_data2.head(30))

#print(data1.head(30))
#print("-"*10)
#print(data_val.head(30))

# Remove outliers based on certain threshold values
data1 = data1.drop(data1[data1['LotArea'] > 100000].index)
data1 = data1.drop(data1[data1['BsmtFinSF1'] > 4000].index)
data1 = data1.drop(data1[data1['TotalBsmtSF'] > 5000].index)
data1 = data1.drop(data1[data1['GrLivArea'] > 4000].index)

# Feature Engineering
###CREATE: Feature Engineering for train and test/validation dataset
#data1['TotalBath'] = data1['FullBath'] + data1['HalfBath']
#data1['TotalPorch'] = data1['OpenPorchSF'] + data1['EnclosedPorch'] + data1['ScreenPorch']

#data_val['TotalBath'] = data_val['FullBath'] + data_val['HalfBath']
#data_val['TotalPorch'] = data_val['OpenPorchSF'] + data_val['EnclosedPorch'] + data_val['ScreenPorch']

data1_correlation = data_val.corr()

print(type(data1))
print(type(data_val))
print(type(data1_correlation))
#sns.heatmap(data_val.corr())
plt.show()

#data1.to_csv('./Data/data1.csv')
#data_val.to_csv('./Data/data_val.csv')