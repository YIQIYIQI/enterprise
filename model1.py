import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('data/train.csv')

labels = data['SalePrice']
col_imp = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt']


regressor = RandomForestRegressor()

regressor.fit(data[col_imp], labels)

pickle.dump(regressor, open('model1.pkl','wb'))
