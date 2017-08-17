import pandas as pd
import numpy as np
import os


os.chdir('/home/Akai/Downloads/ml challenge')

data = pd.read_csv('train.csv')
data = data.drop(labels = [ 'ID'], axis = 1)
data = data.fillna(method = 'bfill', axis = 0)
data = data.fillna(method = 'ffill', axis = 0)
data = data.iloc[:10000000, :]


test = pd.read_csv('test.csv')
test = test.drop(labels = [], axis = 1)
test = test.fillna(method = 'bfill', axis = 0)
test = test.fillna(method = 'ffill', axis = 0)

data.offerid = data.offerid.astype(float)
data.category = data.category.astype(float)
data.merchant = data.merchant.astype(float)
data.click = data.click.astype(float)

uc = data.countrycode.unique()
ordered_c = []
for i in range(len(uc)):
    ordered_c.append(uc[i])
    
ud = data.devid.unique()
ordered_d = []
for i in range(len(ud)):
    ordered_d.append(ud[i])
    
ub = data.browserid.unique()
ordered_b = []
for i in range(len(ub)):
    ordered_b.append(ub[i])
    
data.countrycode = data.countrycode.astype("category", ordered=True, categories=ordered_c).cat.codes
data.devid = data.devid.astype("category", ordered=True, categories=ordered_d).cat.codes
data.browserid = data.browserid.astype("category", ordered=True, categories=ordered_b).cat.codes

data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['weekday'] = data['datetime'].dt.weekday
        
#test_data

test.offerid = test.offerid.astype(float)
test.category = test.category.astype(float)
test.merchant = test.merchant.astype(float)


test.countrycode = test.countrycode.astype("category", ordered=True, categories=ordered_c).cat.codes
test.devid = test.devid.astype("category", ordered=True, categories=ordered_d).cat.codes
test.browserid = test.browserid.astype("category", ordered = True, categories = ordered_b).cat.codes

test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['weekday'] = test['datetime'].dt.weekday

data.to_csv('training_data.csv', index = False)
test.to_csv('test_data.csv', index = False)