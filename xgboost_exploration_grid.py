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

# one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
#                        'heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty',
#                        'regionidneighborhood', 'regionidzip']
# one_hot_encode_cols.extend(set(obj_cols) - set(one_hot_encode_cols))
#
# print 'One Hot Encoding features: %s' % ','.join(one_hot_encode_cols)
# for col in one_hot_encode_cols:
#     one_hot = pd.get_dummies(prop[col], prefix=col)
#     prop = prop.drop(col, axis=1)
#     prop = prop.join(one_hot)


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

xgb_reg = XGBRegressor(eval_metric='mae', early_stopping_rounds=30, n_jobs=4,
                       verbose_eval=10, verbose=10)
xgb_params = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.02, 0.033, 0.1],
    'n_estimators': [350, 500, 1000]
}

xgb_model = xgb.XGBRegressor()
grid = GridSearchCV(xgb_reg, xgb_params, cv=5)
grid.fit(x_train, y_train)
print 'cv_results_', grid.cv_results_
print 'best_score_', grid.best_score_
print 'best_params_', grid.best_params_
pred_train = grid.predict(x_train)

print 'mean_absolute_error', mean_absolute_error(y_train, pred_train)

print ('Building test set...')
sample['parcelid'] = sample['ParcelId']
sample_with_prop = sample.merge(prop, how='left', on='parcelid')


print ('Predicting on test...')
p_test = grid.predict(sample_with_prop[x_train.columns])
sample = sample.drop(['parcelid'], axis=1)
for col in sample.columns:
    if col == 'ParcelId':
        continue
    sample[col] = p_test

print ('Writing to CSV...')
sample.to_csv('output/' + csv_name, index=False, float_format='%.4f')