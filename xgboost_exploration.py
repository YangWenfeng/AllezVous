"""
Explore with XGBoost, by Yang
1. fillna and label encode on object features
2. train with outliers
3. one hot encode
4. gridsearch and cross validation
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5

csv_name = sys.argv[1] if len(sys.argv) >= 2 else 'xgboost_exploration.csv'

print('Reading train data, properties and test data...')
train = pd.read_csv("input/train_2016_v2.csv")
prop = pd.read_csv('input/properties_2016.csv')
sample = pd.read_csv('input/sample_submission.csv')

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
    prop = prop.drop(col, axis=1)
    prop = prop.join(one_hot)


print 'Merge train and properties on parcelid...'
train_with_prop = train.merge(prop, how='left', on='parcelid')

print('Original training data with properties shape: {}'
      .format(train_with_prop.shape))

print('Dropping out outliers.')
train_with_prop = train_with_prop[
    train_with_prop.logerror > OUTLIER_LOWER_BOUND]
train_with_prop = train_with_prop[
    train_with_prop.logerror < OUTLIER_UPPER_BOUND]
print('New training data with properties without outliers shape: {}'
      .format(train_with_prop.shape))

train_drop_cols = ['parcelid', 'logerror', 'transactiondate']
print 'Drop train_with_prop columns: %s' % ','.join(train_drop_cols)
x_train = train_with_prop.drop(train_drop_cols, axis=1)
y_train = train_with_prop['logerror']
d_train = xgb.DMatrix(x_train, y_train)

# xgboost params
params = {'eta': 0.02, 'objective': 'reg:linear', 'eval_metric': 'mae', 'max_depth': 4, 'silent': 1}
estop = 30
cv_res = xgb.cv(params, d_train, num_boost_round=500, early_stopping_rounds=estop, nfold=FOLDS,
                verbose_eval=10, show_stdv=False)

# https://stackoverflow.com/questions/40500638/xgboost-cv-and-best-iteration
# num_boost_rounds = int((cv_res.shape[0] - estop) / (1. - 1. / FOLDS))
num_boost_rounds = int(round(len(cv_res) * np.sqrt(FOLDS/(FOLDS-1.))))
print 'Find num_boost_rounds = %d, cv_res.shape[0] = %d' % (num_boost_rounds, cv_res.shape[0])

model = xgb.train(params, d_train, num_boost_round=num_boost_rounds)
pred_train = model.predict(d_train)

print 'mean_absolute_error', mean_absolute_error(y_train, pred_train)

print ('Building test set...')
sample['parcelid'] = sample['ParcelId']
sample_with_prop = sample.merge(prop, how='left', on='parcelid')

print ('Predicting on test...')
d_test = xgb.DMatrix(sample_with_prop[x_train.columns])
p_test = model.predict(d_test)
sample = sample.drop(['parcelid'], axis=1)
for col in sample.columns:
    if col == 'ParcelId':
        continue
    sample[col] = p_test

print ('Writing to CSV...')
sample.to_csv('output/' + csv_name, index=False, float_format='%.4f')