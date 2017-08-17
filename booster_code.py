import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb

os.chdir('/home/Akai/Downloads/ml challenge')

data = pd.read_csv('training_data.csv')
data = data.drop(labels = ['datetime'], axis = 1)
test = pd.read_csv('test_data.csv')
test_id = test.ID
test = test.drop(labels = ['ID', 'datetime'], axis = 1)


params = {"objective":"binary:logistic",
          "booster":"gbtree",
          "eta":0.1,
          "max_depth": 9,
         "subsample":0.9,
         "cosample_bytree":0.7,
         "silent":0,
         "seed":0,
         "lambda":0.1,
         "alpha":0,
         "eval_metric":"auc"}

num_boost_round = 500

x = data.drop(labels = ['click'], axis =1)
y = data.drop(labels = [ 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'devid', 'hour', 'weekday', 'browserid'], axis = 1)

x_tr = x.iloc[:8000000, :]
y_tr = y.iloc[:8000000, :]

x_val = x.iloc[8000000:, :]
y_val = y.iloc[8000000:, :]

dtrain = xgb.DMatrix(x_tr, y_tr)
dvalid = xgb.DMatrix(x_val, y_val)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

booster = xgb.train(params, dtrain, num_boost_round, evals = watchlist)

dtest = xgb.DMatrix(test)

predictions = booster.predict(dtest)
predicted_values = []
for i in range(len(test)):
    predicted_values.append(predictions[i])
    
submission = pd.DataFrame({'ID':test_id, 'click':predicted_values})
submission.to_csv('submission.csv', index = False)