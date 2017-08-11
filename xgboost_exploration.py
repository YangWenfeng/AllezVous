"""
exploration with XGBoost, by Yang
Function 'load_data' is from Kaggle,
https://www.kaggle.com/c/zillow-prize-1/discussion/37261
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data():
    """Load dataset"""
    train_p = 'input/train_p'
    prop_p = 'input/prop_p'
    sample_p = 'input/sample_p'

    if os.path.exists(train_p):
        train = pd.read_pickle(train_p)
    else:
        train = pd.read_csv('input/train_2016_v2.csv',
                            parse_dates=['transactiondate'])
        train.to_pickle(train_p)

    if os.path.exists(prop_p):
        prop = pd.read_pickle(prop_p)
    else:
        prop = pd.read_csv('input/properties_2016.csv')
        print('Binding to float32')
        for col, dtype in zip(prop.columns, prop.dtypes):
            if dtype == np.float64:
                prop[col] = prop[col].astype(np.float32)
        prop.to_pickle(prop_p)

    if os.path.exists(sample_p):
        sample = pd.read_pickle(sample_p)
    else:
        sample = pd.read_csv('input/sample_submission.csv')
        sample.to_pickle(sample_p)
    return prop, train, sample


print 'Loading data...'
prop, train, sample = load_data()

print 'Processing properties...'
prop_drop_cols = ['propertyzoningdesc', 'propertycountylandusecode', 'latitude', 'longitude']
print 'Drop properties\' columns: %s' % ','.join(prop_drop_cols)
prop = prop.drop(prop_drop_cols, axis=1)

obj_cols = prop.columns[prop.dtypes == object]
print 'Find features with object type: %s' % ','.join(obj_cols)

print 'Encoding and fill missing value on properties\' features with object type...'
for col in obj_cols:
    prop[col] = prop[col].fillna(-1)
    # encode Y class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(prop[col].values[:])
    prop[col] = label_encoder.transform(prop[col].values[:])

one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty',
                       'regionidneighborhood', 'regionidzip']
one_hot_encode_cols.extend(set(obj_cols) - set(one_hot_encode_cols))

print 'One Hot Encoding features: %s' % ','.join(one_hot_encode_cols)
for col in one_hot_encode_cols:
    one_hot = pd.get_dummies(prop[col], prefix=col)
    df = prop.drop(col, axis=1)
    df = prop.join(one_hot)


print 'Merge train and properties on parcelid...'
train_with_prop = train.merge(prop, how='left', on='parcelid')

train_drop_cols = ['parcelid', 'logerror', 'transactiondate']
print 'Drop train_with_prop columns: %s' % ','.join(train_drop_cols)
x_train = train_with_prop.drop(train_drop_cols, axis=1)
y_train = train_with_prop['logerror'].values
d_train = xgb.DMatrix(x_train, y_train)

# xgboost params
params = {'eta': 0.02, 'objective': 'reg:linear', 'eval_metric': 'mae', 'max_depth': 4, 'silent': 1}
estop = 100
FOLDS = 5
cv_res = xgb.cv(params, d_train, num_boost_round=1000, early_stopping_rounds=estop, nfold=FOLDS,
       verbose_eval=10, show_stdv=False)

# https://stackoverflow.com/questions/40500638/xgboost-cv-and-best-iteration
best_nrounds = int((cv_res.shape[0] - estop) / (1 - 1 / FOLDS))
num_boost_rounds = int(round(len(cv_res) * np.sqrt(FOLDS/(FOLDS-1))))
print 'Find best_nrounds = %d, and num_boost_rounds = %d' % (best_nrounds, num_boost_rounds)

model = xgb.train(params, d_train, num_boost_round=num_boost_rounds)

print ('Building test set...')
test = sample.copy()
test['parcelid'] = test['ParcelId']
test_with_prop = sample.merge(prop, how='left', on='parcelid')
d_test = xgb.DMatrix(test_with_prop[x_train.columns])

print ('Predicting on test...')

p_test = model.predict(d_test)
for col in sample.columns[sample.columns != 'ParcelId']:
    test[col] = p_test

print ('Writing to CSV...')
sample.to_csv('output/xgboost_exploration.csv', index=False, float_format='%.4f')

