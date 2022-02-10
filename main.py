# House Price Comparison for Kaggle Learn Users - Submission by jflatnote

# Import necessary packages
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Import data and make DataFrames
# Full data set for the first model
training_data_filepath = ('./Data/train.csv')
train_data = pd.read_csv(training_data_filepath)
#train_data = train_data.dropna(axis=0)
y = train_data.SalePrice
X_features = ['MSSubClass', 'MSZoning', 'LotArea', 'LandContour', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'PavedDrive',	'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
X = pd.get_dummies(train_data[X_features])
X.fillna(X.mean(), inplace=True)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Fits and runs a Decision Tree model at a given depth (max_leaf_nodes) and return an MAE value
def get_mae(nodes, train_X, val_X, train_y, val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict (val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Create a pandas DataFrame to hold the MAE results
mae_results = pd.DataFrame([[0,0],[0,0]], columns = ['max_leaves','mae'])

print(mae_results)

for i in range(5000):
    nodes = i + 2
    my_mae = get_mae(nodes, train_X, val_X, train_y, val_y)
    mae_results.loc[i] = [nodes,my_mae]

print(mae_results.head())
print(mae_results[mae_results.mae == mae_results.mae.min()])